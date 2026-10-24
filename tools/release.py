#!/usr/bin/env python3
"""Verify the source manifest or build a source-only ZIP from an explicit layout."""

import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PACKAGE_MANIFEST.json"
ROOT_FILES = """
.gitignore AGENTS.md README.md SOURCE_PROVENANCE.json THIRD_PARTY_NOTICES.txt
VALIDATION.json app.py archive_results.py core.py geometry.py job.sh lf.py
mapper.py models.py mpnn_helpers.py objective.py prepare.py requirements.txt
requirements-ui.txt run120.py search.py site.example.json
""".split()
ASSET_FILES = """
assets/alleles25.json assets/protein_mpnn.pt
examples/luxsit_seed.pdb examples/luxsit_fixed.txt
native/build.sh native/kernel.c native/kernel.so
vendor/__init__.py vendor/mpnn/__init__.py vendor/mpnn/data_utils.py
vendor/mpnn/model_utils.py vendor/af3/LICENSE vendor/af3/Apache-2.0.txt vendor/af3/af3_apptainer.py
vendor/af3/af3_daemon.py vendor/af3/af3_runtime.py
vendor/af3/run_alphafold_batch.py
""".split()
SUPPORT_FILES = """
configs/digs.example.json configs/linux.example.json
docs/AF3.md docs/INSTALL.md docs/PORTING.json docs/TESTING.md docs/WORKFLOW.md docs/SCORING_REVISION.json
docs/readability-check.json docs/reference/PACKAGE_MANIFEST.upstream.json
docs/reference/README.upstream.md docs/reference/campaign_prepare.upstream.txt
runtime/af3/PROVENANCE.json runtime/af3/README.md
runtime/af3/af3-open-2026-08-19.def runtime/af3/af3-open-2026-08-19.labels.txt
tests/cpu_smoke.py tests/reference_policy.py tests/test_campaign.py
tests/test_cleanup.py tests/test_packaging.py tests/test_portable.py tests/test_weighted.py
tools/release.py
""".split()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def payload():
    names = set(ROOT_FILES + ASSET_FILES + SUPPORT_FILES)
    files = {}
    for name in sorted(names):
        path = ROOT / name
        if not path.is_file():
            raise ValueError("Missing source file: " + name)
        if any(p.is_symlink() for p in (path, *path.parents) if p != ROOT):
            raise ValueError("Release files must not be symlinks: " + name)
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("Release file escapes package: " + name)
        data = path.read_bytes()
        files[name] = {"sha256": digest(data), "bytes": len(data)}
    return files


def verify(files):
    recorded = json.loads(MANIFEST.read_text())["files"]
    changed = [
        name
        for name in sorted(set(files) | set(recorded))
        if files.get(name) != recorded.get(name)
    ]
    if changed:
        raise ValueError(
            "Source manifest differs; review changes before --refresh:\n"
            + "\n".join(changed)
        )


def archive(output, files):
    output = output.expanduser().resolve()
    if output.suffix != ".zip":
        raise ValueError("Output must have a .zip extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as bundle:
        for name in sorted([*files, MANIFEST.name]):
            path = ROOT / name
            info = zipfile.ZipInfo(
                "FastSearchAF3/" + name, date_time=(2026, 10, 18, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (
                stat.S_IFREG | stat.S_IMODE(path.stat().st_mode)
            ) << 16
            bundle.writestr(info, path.read_bytes())
    checksum = digest(output.read_bytes())
    output.with_suffix(".zip.sha256").write_text(checksum + "  " + output.name + "\n")
    print(f"Created {output} ({output.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Verify all released source hashes"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Record a new manifest after reviewing deliberate edits",
    )
    parser.add_argument(
        "--output", type=Path, help="Build a ZIP only after manifest verification"
    )
    args = parser.parse_args()
    if not (args.check or args.refresh or args.output):
        parser.error("Choose --check, --refresh, or --output")
    files = payload()
    if args.refresh:
        MANIFEST.write_text(
            json.dumps(
                {
                    "package": "FastSearchAF3",
                    "packaging_version": "2026.10.18",
                    "files": files,
                },
                indent=2,
            )
            + "\n"
        )
    verify(files)
    print(f"Verified {len(files)} source files")
    if args.output:
        archive(args.output, files)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
