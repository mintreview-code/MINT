"""Arbitrary-length single-chain PDB normalization with explicit residue mapping."""

import math
from pathlib import Path
import re

AA = "ACDEFGHIKLMNPQRSTVWY"
AA3 = dict(
    zip(
        AA,
        "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    )
)
FIXED = set()


def configure(config):
    global FIXED
    # Configuration uses one-based residue indices; string indexing is zero-based.
    FIXED = {int(i) - 1 for i in config["fixed_positions"]}


def normalize(path, chain=None):
    """Select one canonical protein chain, retain N/CA/C/O, renumber A:1..L.

    Reject missing backbone atoms and physical chain breaks rather than inventing
    peptide windows across unresolved sequence. Ignore waters/ligands explicitly.
    """
    residues = {}
    order = []
    model_seen = False
    for line in Path(path).read_text().splitlines():
        if line.startswith("MODEL"):
            if model_seen:
                break
            model_seen = True
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  "):
            continue
        if len(line) < 54:
            raise ValueError("Malformed ATOM record")
        atom = line[12:16].strip()
        if atom not in ("N", "CA", "C", "O"):
            continue
        key = (line[21], int(line[22:26]), line[26].strip())
        if key not in residues:
            residues[key] = {}
            order.append(key)
        alt = line[16]
        occ = float(line[54:60].strip() or 0)
        priority = (alt == " ", alt == "A", occ)
        if atom not in residues[key] or priority > residues[key][atom][0]:
            residues[key][atom] = (priority, line)
    chains = list(dict.fromkeys(k[0] for k in order))
    if not chains:
        raise ValueError("No protein ATOM backbone found")
    if chain is None:
        if len(chains) != 1:
            raise ValueError("Multiple chains found; specify --chain: " + repr(chains))
        chain = chains[0]
    if chain not in chains:
        raise ValueError("Selected chain absent: " + repr(chain))
    selected = [k for k in order if k[0] == chain]
    lines = []
    mapping = []
    sequence = []
    previous = None
    serial = 0
    if not 11 <= len(selected) <= 9999:
        raise ValueError(
            "Selected chain must contain 11..9999 residues (AF3/GPU limits can be lower)"
        )
    for index, key in enumerate(selected, 1):
        atoms = residues[key]
        if set(atoms) != set(("N", "CA", "C", "O")):
            raise ValueError("Missing backbone atoms at " + str(key))
        names = {x[1][17:20] for x in atoms.values()}
        if len(names) != 1 or next(iter(names)) not in set(AA3.values()):
            raise ValueError("Noncanonical or inconsistent residue at " + str(key))
        name = next(iter(names))
        sequence.append({v: k for k, v in AA3.items()}[name])

        def xyz(line):
            return [float(line[30:38]), float(line[38:46]), float(line[46:54])]

        current = xyz(atoms["N"][1])
        if previous is not None:
            separation = math.sqrt(sum((a - b) ** 2 for a, b in zip(previous, current)))
            if not 0.5 <= separation <= 2.2:
                raise ValueError(
                    "Backbone chain break before "
                    + str(key)
                    + "; supply a complete contiguous chain"
                )
        previous = xyz(atoms["C"][1])
        mapping.append(
            dict(
                index=index,
                chain=key[0],
                residue_number=key[1],
                insertion_code=key[2],
                amino_acid=sequence[-1],
                original_id="%s:%d%s" % key,
            )
        )
        for atom in ("N", "CA", "C", "O"):
            serial += 1
            line = atoms[atom][1]
            coords = xyz(line)
            if not all(math.isfinite(x) for x in coords):
                raise ValueError("Nonfinite coordinates")
            lines.append(
                line[:6]
                + f"{serial:5d}"
                + line[11:16]
                + " "
                + line[17:21]
                + "A"
                + f"{index:4d}"
                + " "
                + line[27:]
            )
    return "\n".join(lines) + "\nEND\n", "".join(sequence), mapping


def fixed_positions(tokens, mapping, indices=""):
    byid = {r["original_id"]: r["index"] for r in mapping}
    chain = mapping[0]["chain"]
    out = set()
    for token in re.split(r"[\s,]+", tokens.strip()):
        if not token or token.startswith("#"):
            continue
        if ":" not in token:
            match = re.fullmatch(r"([A-Za-z])(-?\d+[A-Za-z]?)", token)
            token = (match[1] + ":" + match[2]) if match else chain + ":" + token
        if token not in byid:
            raise ValueError("Unknown fixed PDB residue " + token)
        out.add(byid[token])
    for token in re.split(r"[\s,]+", indices.strip()):
        if not token:
            continue
        if "-" in token:
            a, b = map(int, token.split("-"))
            out.update(range(a, b + 1))
        else:
            out.add(int(token))
    if not out <= set(range(1, len(mapping) + 1)):
        raise ValueError("Fixed index outside the selected chain")
    if len(out) == len(mapping):
        raise ValueError("At least one residue must be mutable")
    return sorted(out)


def pdb_data(path):
    sequence = []
    coords = []
    keys = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[16] in " A":
            key = (line[21], int(line[22:26]), line[26].strip())
            if key in keys:
                continue
            keys.append(key)
            sequence.append({v: k for k, v in AA3.items()}[line[17:20]])
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if keys != [("A", i, "") for i in range(1, len(keys) + 1)] or not keys:
        raise ValueError("Expected normalized chain A with contiguous indices")
    return "".join(sequence), coords


def invariant(sequence, root):
    if len(sequence) != len(root) or set(sequence) - set(AA):
        raise ValueError("Sequence length/alphabet changed")
    if any(sequence[i] != root[i] for i in FIXED):
        raise ValueError("Fixed residue identity changed")


def write_design_pdb(parent, sequence, destination):
    original, _ = pdb_data(parent)
    if len(sequence) != len(original):
        raise ValueError("Sequence length changed")
    lines = []
    for line in Path(parent).read_text().splitlines():
        if (
            line.startswith("ATOM")
            and line[21] == "A"
            and line[12:16].strip() in ("N", "CA", "C", "O")
        ):
            i = int(line[22:26]) - 1
            lines.append(line[:17] + AA3[sequence[i]] + line[20:])
    if len(lines) != 4 * len(sequence):
        raise ValueError("Incomplete backbone")
    Path(destination).write_text("\n".join(lines) + "\nEND\n")


def rmsd(moving, reference):
    import numpy as np

    p = np.asarray(moving, dtype=float)
    q = np.asarray(reference, dtype=float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("Coordinate shape mismatch")
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.eye(3)
    d[2, 2] = np.linalg.det(u @ vt)
    value = float(np.sqrt(np.mean(np.sum((p @ u @ d @ vt - q) ** 2, axis=1))))
    if not math.isfinite(value):
        raise ValueError("Nonfinite RMSD")
    return value
