"""Resident CPU MPNN and the installed, warm single-GPU AF3 socket adapter."""

from core import *
import io
import socket
import tempfile


class MPNN:
    def __init__(self, config, scratch):
        import torch

        self.torch = torch
        torch.set_num_threads(1)
        self.config = config
        self.scratch = Path(scratch)
        self.g = load("portable_mpnn_helpers", HERE / "mpnn_helpers.py")
        self.device = torch.device("cpu")
        self.model = self.g.build_model(
            str(HERE / "assets/protein_mpnn.pt"), self.device
        )
        self.features = {}

    def setup(self, parent):
        key = sha(parent["sequence"] + parent["pdb"])
        if key not in self.features:
            path = self.scratch / "mpnn_parent.pdb"
            path.write_text(parent["pdb"])
            sequence, _ = geometry.pdb_data(path)
            if sequence != parent["sequence"]:
                raise ValueError("Parent structure/sequence mismatch")
            fd, *_ = self.g.setup_protein(
                str(path),
                ["A%d" % p for p in sorted(m.FIXED)],
                self.config["alleles"],
                self.device,
            )
            fd, _ = self.g.base_feature_common(fd, 1.0, {"C", "X"}, self.device)
            if len(self.features) >= 32:
                self.features.clear()
            self.features[key] = fd
        # Features mutate during sampling; rebuild sequence/mask on EVERY call.
        return dict(self.features[key])

    def sample(self, parent, positions, temperature, seed, from_scratch=False):
        torch = self.torch
        fd = self.setup(parent)
        positions = sorted(set(positions))
        if not set(positions) <= set(range(1, len(parent["sequence"]) + 1)) - m.FIXED:
            raise ValueError("Fixed MPNN position")
        torch.manual_seed(seed)
        fd["S"] = torch.tensor(
            [[m.AA.index(a) for a in parent["sequence"]]], device=self.device
        )
        fd["chain_mask"] = torch.zeros((1, len(parent["sequence"])), device=self.device)
        for p in positions:
            fd["chain_mask"][0, p - 1] = 1
        fd["temperature"] = temperature
        fd["init_redesign_w_input"] = not from_scratch
        fd["randn"] = torch.randn((1, len(parent["sequence"])), device=self.device)
        with torch.no_grad():
            result = self.model.sample(fd)
        sequence = "".join((m.AA + "X")[int(x)] for x in result["S"][0])
        geometry.invariant(sequence, self.config["root_sequence"])
        if any(
            sequence[i] != parent["sequence"][i]
            for i in range(len(parent["sequence"]))
            if i + 1 not in positions
        ):
            raise ValueError("MPNN changed unselected site")
        probs = result["sampling_probs"][0, :, :20].cpu().tolist()
        return sequence, probs


class AF3:
    def __init__(self, config, scratch):
        self.config = config
        self.scratch = Path(scratch)
        self.reference = geometry.pdb_data(config["root_pdb"])[1]
        self.wall_seconds = 0.0
        self.driver_cpu_seconds = 0.0
        self.calls = 0

    def request(self, payload):
        # The search process sends jobs to a separate, already-running AF3
        # process. This socket keeps the GPU model resident between batches.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(3600)
            sock.connect(os.environ["FRESH_AF3_SOCKET"])
            sock.sendall((json.dumps(payload) + "\n").encode())
            with sock.makefile("rb") as stream:
                reply = json.loads(stream.readline())
        if reply.get("status") != "ok":
            raise RuntimeError("AF3 failed: " + repr(reply))

    def fold(self, records, seed):
        if not records:
            return []
        from Bio.PDB import PDBParser, MMCIFIO
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict

        before = cpu()
        wall = time.monotonic()
        out = []
        try:
            # Small manifest chunks limit both temporary files and checkpoint loss.
            with tempfile.TemporaryDirectory(dir=self.scratch, prefix="fold_") as temp:
                temp = Path(temp)
                payload = []
                for i, r in enumerate(records):
                    name = "s" + sha(r["sequence"])[:16] + "_" + str(seed + i)
                    parent = temp / "parent.pdb"
                    parent.write_text(r["pdb"])
                    design = temp / "design.pdb"
                    # Use the parent backbone to serialize the candidate as CIF.
                    # The request has no template field, so the runtime clears
                    # templates; these coordinates do not constrain the AF3 fold.
                    geometry.write_design_pdb(parent, r["sequence"], design)
                    structure = PDBParser(QUIET=True).get_structure(name, str(design))
                    writer = MMCIFIO()
                    writer.set_structure(structure)
                    stream = io.StringIO()
                    writer.save(stream)
                    data = MMCIF2Dict(io.StringIO(stream.getvalue()))
                    # Bio's atom-only CIF omits polymer identity tables required
                    # by AF3; explicitly supply the single canonical chain.
                    data.update(
                        {
                            "_entry.id": name,
                            "_entity.id": ["1"],
                            "_entity.type": ["polymer"],
                            "_struct_asym.id": ["A"],
                            "_struct_asym.entity_id": ["1"],
                            "_entity_poly.entity_id": ["1"],
                            "_entity_poly.type": ["polypeptide(L)"],
                            "_entity_poly.pdbx_seq_one_letter_code": [r["sequence"]],
                            "_entity_poly.pdbx_seq_one_letter_code_can": [
                                r["sequence"]
                            ],
                            "_entity_poly.pdbx_strand_id": ["A"],
                            "_entity_poly_seq.entity_id": ["1"] * len(r["sequence"]),
                            "_entity_poly_seq.num": [
                                str(j) for j in range(1, len(r["sequence"]) + 1)
                            ],
                            "_entity_poly_seq.mon_id": [
                                geometry.AA3[a] for a in r["sequence"]
                            ],
                            "_entity_poly_seq.hetero": ["n"] * len(r["sequence"]),
                        }
                    )
                    data["_atom_site.label_entity_id"] = ["1"] * len(
                        data["_atom_site.id"]
                    )
                    writer.set_dict(data)
                    stream = io.StringIO()
                    writer.save(stream)
                    payload.append(
                        dict(
                            name=name,
                            cif=stream.getvalue(),
                            output_dir=str(temp / name),
                            seed=seed + i,
                            paired_msa={"A": ""},
                            unpaired_msa={"A": ""},
                        )
                    )
                manifest = temp / "manifest.json"
                save(manifest, payload)
                self.request(
                    dict(cmd="fold", manifest=str(manifest), save_full_outputs=False)
                )
                for r, p in zip(records, payload):
                    output = Path(p["output_dir"])
                    metrics, pdb = parse_fold(
                        output / (p["name"] + "_model.cif"),
                        r["sequence"],
                        self.config["root_sequence"],
                        self.reference,
                    )
                    report = {k: v for k, v in r.items() if k != "pdb"}
                    report.update(
                        metrics,
                        af3_seed=p["seed"],
                        af3_recycles=5,
                        af3_diffusion_samples=1,
                        confidence_summary=read(
                            output / (p["name"] + "_summary_confidences.json")
                        ),
                    )
                    if report["passes"]:
                        report["pdb"] = pdb
                    out.append(report)
                    self.calls += 1
            return out
        finally:
            self.wall_seconds += time.monotonic() - wall
            self.driver_cpu_seconds += cpu() - before


def parse_fold(path, expected, root_sequence, reference):
    from Bio.PDB import MMCIFParser, PDBIO

    structure = MMCIFParser(QUIET=True).get_structure("af3", str(path))
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError("Expected one AF3 model")
    chains = list(models[0].get_chains())
    if len(chains) != 1 or chains[0].id != "A":
        raise ValueError("Unexpected chains")
    residues = list(chains[0].get_residues())
    if [r.id for r in residues] != [(" ", i, " ") for i in range(1, len(expected) + 1)]:
        raise ValueError("Unexpected residue mapping")
    names = {v: k for k, v in geometry.AA3.items()}
    seq = "".join(names[r.resname] for r in residues)
    if seq != expected:
        raise ValueError("AF3 sequence changed")
    geometry.invariant(seq, root_sequence)
    scores = [float(r["CA"].bfactor) for r in residues]
    if any(not math.isfinite(x) or not 0 <= x <= 100 for x in scores):
        raise ValueError("Invalid pLDDT")
    coords = [r["CA"].coord.tolist() for r in residues]
    rmsd = geometry.rmsd(coords, reference)
    plddt = sum(scores) / len(scores)
    writer = PDBIO()
    writer.set_structure(structure)
    stream = io.StringIO()
    writer.save(stream)
    return (
        dict(
            passes=plddt >= 90 and rmsd <= 2,
            plddt=plddt,
            root_ca_rmsd=rmsd,
            per_residue_plddt=scores,
            structure_sha256=m.digest(path),
        ),
        stream.getvalue(),
    )
