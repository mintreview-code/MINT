#!/usr/bin/env python3
"""Portable LF entry point. Preparing/checking a run never submits it."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geometry
import mapper


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def site_settings(path=None):
    config = read(HERE / "site.example.json")
    if path:
        config.update(read(path))
    for key in (
        "cpu_container",
        "cpu_python",
        "netmhcpan",
        "af3_image",
        "af3_model_dir",
    ):
        if config.get(key):
            config[key] = str(
                Path(os.path.expandvars(os.path.expanduser(config[key]))).absolute()
            )
    if not 2 <= int(config["cpus"]) <= 128:
        raise ValueError("Use at least two CPUs; default is eight")
    if not config["cpu_python"] and not config["cpu_container"]:
        raise ValueError("Provide a CPU container or CPU Python environment")
    return config


def check_site(site):
    paths = {
        "NetMHCpan": Path(site["netmhcpan"]) / "Linux_x86_64/bin/netMHCpan",
        "AF3 image": Path(site["af3_image"]),
        "AF3 models": Path(site["af3_model_dir"]),
        "CPU environment": Path(site["cpu_python"] or site["cpu_container"]),
    }
    missing = [
        name + ": " + str(p)
        for name, p in paths.items()
        if not p.exists() or not os.access(p, os.R_OK)
    ]
    if missing:
        raise ValueError("Unavailable dependencies:\n" + "\n".join(missing))
    return {name: str(p) for name, p in paths.items()}


def protected_sources():
    paths = list(HERE.glob("*.py")) + [HERE / "job.sh"]
    paths += list((HERE / "vendor").rglob("*.py"))
    paths += [
        HERE / "native/kernel.so",
        HERE / "native/kernel.c",
        HERE / "assets/protein_mpnn.pt",
    ]
    return {str(p.relative_to(HERE)): mapper.digest(p) for p in paths}


def initialize(args):
    out = args.out.expanduser().resolve()
    if out.exists():
        raise ValueError("Output already exists; choose a new run directory")
    site = site_settings(args.site_config)
    check_site(site)
    text, sequence, mapping = geometry.normalize(args.pdb, args.chain)
    tokens = args.fixed or ""
    if args.fixed_file:
        tokens += " " + " ".join(
            line.split("#", 1)[0] for line in args.fixed_file.read_text().splitlines()
        )
    fixed = geometry.fixed_positions(tokens, mapping, args.fixed_indices or "")
    alleles = (
        read(args.alleles_file)
        if args.alleles_file
        else read(HERE / "assets/alleles25.json")
    )
    if (
        not isinstance(alleles, list)
        or not alleles
        or any(
            not isinstance(a, str) or "," in a or not a.startswith("HLA-")
            for a in alleles
        )
    ):
        raise ValueError("Alleles file must be a JSON list of HLA allele names")
    alleles = [a.replace("*", "") for a in alleles]
    if len(set(alleles)) != len(alleles):
        raise ValueError("Duplicate alleles")
    config = dict(
        direction=args.direction,
        objective=(
            "cumulative rank-count sum"
            if getattr(args, "objective_mode", "weighted8") == "weighted8"
            else "N(EL %Rank <2%)"
        ),
        objective_mode=getattr(args, "objective_mode", "weighted8"),
        alleles=alleles,
        root_pdb="input.pdb",
        root_sequence=sequence,
        fixed_positions=fixed,
        seed=args.seed,
        temperatures=[0.4],
        initial_designs=args.initial_designs,
        initial_folds=args.initial_folds,
        rounds=args.rounds,
        beam=args.beam,
        parents_per_round=min(8, args.beam),
        max_positions=len(sequence) - len(fixed),
        position_pairs=8,
        joint_mpnn_per_parent=8,
        folds_per_round=args.folds_per_round,
        immigrant_interval=5,
        immigrants=16,
        worsening_fraction=0.02,
        predictor_workers=site["cpus"] - 1,
        cpus=site["cpus"],
        gpus=1,
        search_hours=args.search_hours,
        af3_recycles=5,
        af3_samples=1,
        plddt_gate=90,
        rmsd_gate=2,
        native_mode=args.native_mode,
        netmhcpan=site["netmhcpan"],
        site=site,
        protected_sources=protected_sources(),
        new_prediction_cache_required=True,
        no_stepwise_threshold_optimization=True,
        creation_time=time.time(),
        package_version="1.0.0",
        source_input_sha256=mapper.digest(args.pdb),
        input_sha256=mapper.sha(text.encode()),
    )
    if args.smoke:
        config.update(
            initial_designs=4,
            initial_folds=5,
            rounds=1,
            beam=2,
            parents_per_round=1,
            max_positions=min(2, len(sequence) - len(fixed)),
            position_pairs=1,
            joint_mpnn_per_parent=2,
            folds_per_round=2,
            immigrants=0,
            search_hours=1.5,
        )
    for key in ("rounds", "beam", "initial_folds", "folds_per_round"):
        if config[key] < 1:
            raise ValueError(key + " must be positive")
    if config["initial_designs"] < 0 or config["search_hours"] <= 0:
        raise ValueError("Invalid design/time budget")
    if args.cleanup_scratch_on_success and (
        not args.scratch_root or not args.archive_dir
    ):
        raise ValueError(
            "--cleanup-scratch-on-success requires --scratch-root and --archive-dir"
        )
    if not args.cleanup_scratch_on_success and (args.scratch_root or args.archive_dir):
        raise ValueError("Archive paths require --cleanup-scratch-on-success")
    out.mkdir(parents=True)
    config["cleanup_scratch_on_success"] = False
    if args.cleanup_scratch_on_success:
        import archive_results

        config.update(
            archive_results.initialize(out, args.scratch_root, args.archive_dir)
        )
    (out / "input.pdb").write_text(text)
    save(out / "residue_mapping.json", mapping)
    save(out / "config.json", config)
    (out / "fixed_residues.txt").write_text(
        "\n".join(r["original_id"] for r in mapping if r["index"] in fixed) + "\n"
    )
    print("Prepared " + str(out))
    print(
        "Direction: %s; length: %d; fixed: %d; alleles: %d"
        % (args.direction, len(sequence), len(fixed), len(alleles))
    )
    return out


def submission_command(run):
    c = read(run / "config.json")
    s = c["site"]
    check_site(s)
    for path, value in c["protected_sources"].items():
        if mapper.digest(HERE / path) != value:
            raise ValueError("Package source changed after run preparation: " + path)
    if mapper.digest(run / c["root_pdb"]) != c["input_sha256"]:
        raise ValueError("Input PDB changed")
    cmd = [
        "sbatch",
        "--parsable",
        "--partition=" + s["partition"],
        "--gres=" + s["gres"],
        "--cpus-per-task=" + str(s["cpus"]),
        "--mem=" + s["memory"],
        "--time=" + s["time_limit"],
        "--job-name=lfp_" + c["direction"],
        "--signal=B:USR1@300",
        "--output=" + str(run / "slurm-%j.log"),
    ]
    if s.get("constraint"):
        cmd += ["--constraint=" + s["constraint"]]
    if s.get("account"):
        cmd += ["--account=" + s["account"]]
    return cmd + [str(HERE / "job.sh"), str(run)]


def submit(run, dry=False, resume=False):
    run = run.resolve()
    cmd = submission_command(run)
    if dry:
        print(shlex.join(cmd))
        return
    previous = run / "submission.json"
    if previous.exists():
        if not resume:
            raise ValueError(
                "Already submitted; --resume is required to restart an interrupted run"
            )
        if (run / "complete.json").exists():
            raise ValueError("Run already completed its configured budget")
        if not (run / "state.json.gz").exists():
            raise ValueError("No resume checkpoint available")
        active = subprocess.run(
            ["squeue", "-h", "-j", read(previous)["job"], "-o", "%T"],
            check=True,
            text=True,
            capture_output=True,
        )
        if active.stdout.strip():
            raise ValueError("Previous job is still active")
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError("Unexpected sbatch reply: " + result.stdout)
    record = dict(job=job, command=cmd, time=time.time(), resume=resume)
    save(previous, record)
    with (run / "submissions.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print("Submitted " + job)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for action in ("init", "launch"):
        q = sub.add_parser(action)
        q.add_argument("--pdb", type=Path, required=True)
        q.add_argument("--out", type=Path, required=True)
        q.add_argument("--direction", choices=["deimmunize", "immunize"], required=True)
        q.add_argument("--chain")
        q.add_argument(
            "--fixed", help="Original PDB IDs: A:10,A:20A (legacy A10 also accepted)"
        )
        q.add_argument("--fixed-file", type=Path)
        q.add_argument(
            "--fixed-indices", help="One-based selected-chain indices, e.g. 1,6,9-10"
        )
        q.add_argument("--alleles-file", type=Path)
        q.add_argument("--site-config", type=Path)
        q.add_argument("--rounds", type=int, default=80)
        q.add_argument("--search-hours", type=float, default=44)
        q.add_argument(
            "--objective-mode", choices=["broad2", "weighted8"], default="weighted8"
        )
        q.add_argument("--initial-designs", type=int, default=512)
        q.add_argument("--initial-folds", type=int, default=96)
        q.add_argument("--folds-per-round", type=int, default=48)
        q.add_argument("--beam", type=int, default=24)
        q.add_argument("--seed", type=int, default=26051011)
        q.add_argument(
            "--native-mode", choices=["auto", "required", "off"], default="auto"
        )
        q.add_argument(
            "--smoke",
            action="store_true",
            help="Small real-model integration run, including AF3",
        )
        q.add_argument(
            "--cleanup-scratch-on-success",
            action="store_true",
            help="Verify a permanent result archive, then delete this scratch run and cache on successful completion",
        )
        q.add_argument(
            "--scratch-root",
            type=Path,
            help="Explicit parent directory containing this scratch run",
        )
        q.add_argument(
            "--archive-dir",
            type=Path,
            help="New permanent result directory outside scratch",
        )
    q = sub.add_parser("submit")
    q.add_argument("--run", type=Path, required=True)
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--resume", action="store_true")
    q = sub.add_parser("doctor")
    q.add_argument("--site-config", type=Path)
    q.add_argument("--inside", action="store_true")
    a = p.parse_args()
    if a.action in ("init", "launch"):
        run = initialize(a)
        if a.action == "launch":
            submit(run)
    elif a.action == "submit":
        submit(a.run, a.dry_run, a.resume)
    else:
        print(json.dumps(check_site(site_settings(a.site_config)), indent=2))
        if a.inside:
            import torch, numpy, prody, Bio, matplotlib

            print(
                json.dumps(
                    dict(
                        python=sys.version,
                        torch=torch.__version__,
                        numpy=numpy.__version__,
                        prody=prody.__version__,
                        biopython=Bio.__version__,
                        matplotlib=matplotlib.__version__,
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc))
