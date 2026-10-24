#!/usr/bin/env python3
"""Prepare one run with the revised weighted-score settings and 120-round budget; never submit."""

import argparse
from pathlib import Path

import lf

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE / "examples/luxsit_seed.pdb"


def prepare_run(
    out,
    site_config,
    *,
    pdb=EXAMPLE,
    direction="deimmunize",
    chain=None,
    fixed_file=None,
    fixed="",
    fixed_indices="",
    seed=None,
    native_mode="required",
    smoke=False,
):
    """Adapt the upstream campaign preparation to one portable, independent run.

    Only the bundled LuxSit example gets the bundled 43 fixed positions by
    default. Custom structures require their own explicit fixed-position input.
    """
    pdb = Path(pdb).expanduser().resolve()
    if pdb == EXAMPLE and not (fixed_file or fixed or fixed_indices):
        fixed_file = HERE / "examples/luxsit_fixed.txt"
    args = argparse.Namespace(
        out=Path(out),
        pdb=pdb,
        site_config=Path(site_config).expanduser(),
        direction=direction,
        objective_mode="weighted8",
        chain=chain or None,
        fixed_file=Path(fixed_file).expanduser() if fixed_file else None,
        fixed=fixed,
        fixed_indices=fixed_indices,
        alleles_file=None,
        seed=seed if seed is not None else (202610071 if smoke else 26051011),
        rounds=1 if smoke else 120,
        search_hours=1.5 if smoke else 42,
        initial_designs=4 if smoke else 512,
        initial_folds=5 if smoke else 96,
        folds_per_round=2 if smoke else 48,
        beam=2 if smoke else 24,
        native_mode=native_mode,
        smoke=smoke,
        # Keep runs unless archival/deletion is explicitly configured through lf.py.
        cleanup_scratch_on_success=False,
        scratch_root=None,
        archive_dir=None,
    )
    run = lf.initialize(args)
    config = lf.read(run / "config.json")
    # Initial/immigrant temperatures and budgets retain the original campaign values.
    # The user-approved revision uses temperature 0.4 for every joint proposal.
    config.update(
        initial_temperatures=[0.4],
        immigrant_temperatures=[0.4],
        temperatures=[0.4],
        allocation_search_hours=1.5 if smoke else 42,
        campaign_preset="weighted8_t04_20261018",
        packaging_version="2026.10.18",
    )
    lf.save(run / "config.json", config)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--site-config", required=True, type=Path)
    parser.add_argument("--pdb", type=Path, default=EXAMPLE)
    parser.add_argument("--chain")
    parser.add_argument(
        "--direction", choices=["deimmunize", "immunize"], default="deimmunize"
    )
    parser.add_argument("--fixed", default="", help="Original PDB IDs, e.g. B:40,B:45A")
    parser.add_argument("--fixed-file", type=Path)
    parser.add_argument(
        "--fixed-indices", default="", help="One-based selected-chain indices"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--native-mode", choices=["auto", "required", "off"], default="required"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Original small real-model integration preset",
    )
    args = parser.parse_args()
    run = prepare_run(**vars(args))
    print("Campaign config: " + str(run / "config.json"))
    print(
        "Prepared only. Review the config, then submit or launch on your Linux GPU machine."
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc))
