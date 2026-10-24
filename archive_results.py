"""Opt-in, verified result export followed by deletion of an owned scratch run."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from objective import THRESHOLDS

OWNER = ".scratch_owner.json"
MANIFEST = "archive_manifest.json"


def read(path):
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as handle:
        return json.load(handle)


def save(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inside(path, root):
    return path == root or root in path.parents


def initialize(run, scratch_root, archive):
    run, scratch_root, archive = (
        Path(p).resolve() for p in (run, scratch_root, archive)
    )
    if run.parent != scratch_root:
        raise ValueError(
            "Cleanup requires a single run directly under the explicit scratch root"
        )
    if inside(archive, scratch_root) or inside(run, archive):
        raise ValueError(
            "The permanent archive must be outside scratch and outside the run ancestry"
        )
    if archive.exists():
        raise ValueError("Choose a new permanent archive directory")
    owner = dict(
        token=uuid.uuid4().hex,
        run=str(run),
        scratch_root=str(scratch_root),
        archive=str(archive),
    )
    save(run / OWNER, owner)
    return dict(
        cleanup_scratch_on_success=True,
        scratch_root=str(scratch_root),
        archive_directory=str(archive),
        cleanup_owner_token=owner["token"],
    )


def excluded(relative):
    # The database is rebuildable. Keep provenance and all scientific results.
    return any(
        part == "cache.sqlite"
        or part.startswith(("cache.sqlite-", "cache.sqlite."))
        or part.endswith(".tmp")
        or ".tmp." in part
        for part in relative.parts
    )


def source_files(run):
    result = []
    for folder, directories, files in os.walk(run, followlinks=False):
        for name in directories + files:
            path = Path(folder) / name
            if path.is_symlink():
                raise ValueError(
                    "Refusing scratch cleanup containing a symlink: " + str(path)
                )
        for name in files:
            path = Path(folder) / name
            if not excluded(path.relative_to(run)):
                if not path.is_file():
                    raise ValueError("Unexpected non-regular file: " + str(path))
                result.append(path)
    return sorted(result)


def terminal_success(run, config):
    path = run / "complete.json"
    if not path.exists():
        return False
    done = read(path)
    if not done.get("complete") or done.get("stopped_at_budget"):
        return False
    if not done.get("original_predictor_audit"):
        return False
    if done.get("direction") != config["direction"]:
        raise ValueError("Completion direction does not match the run")
    confirmation = read(run / "confirmation.json.gz")
    if len(confirmation.get("records", [])) != 2 or done.get("confirmation_total") != 2:
        return False
    full_rounds = done["rounds"] >= config["rounds"]
    sequence_length = len(config["root_sequence"])
    maximum = sum(max(0, sequence_length - n + 1) for n in (9, 10, 11)) * len(
        config["alleles"]
    )
    if config.get("objective_mode", "broad2") == "weighted8":
        if done.get("objective_mode") != "weighted8":
            raise ValueError("Completion objective does not match the run")
        score = sum(done["cutoff_counts"][str(t)] for t in THRESHOLDS)
        if done.get("objective_score") != score:
            raise ValueError("Completion weighted score does not match its counts")
        endpoint = score == (0 if config["direction"] == "deimmunize" else 8 * maximum)
    else:
        endpoint = done["counts"][2] == (
            0 if config["direction"] == "deimmunize" else maximum
        )
    verified_endpoint = (
        endpoint
        and done.get("confirmation_passes") == 2
        and all(r["passes"] for r in confirmation["records"])
    )
    if not (full_rounds or verified_endpoint):
        return False
    for name in (
        "state.json.gz",
        "champion.pdb",
        "champion.json",
        "progress.csv",
        "config.json",
        "input.pdb",
        "provenance.json",
    ):
        if not (run / name).is_file():
            raise ValueError("Missing scientific result required for cleanup: " + name)
    if read(run / "state.json.gz")["round"] != done["rounds"]:
        raise ValueError("Checkpoint and completion round disagree")
    return True


def verify_archive(archive, owner):
    manifest = read(archive / MANIFEST)
    if manifest["owner"] != owner:
        raise ValueError("Archive ownership mismatch")
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe archived path")
        path = archive / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row["bytes"]
            or digest(path) != row["sha256"]
        ):
            raise ValueError("Archive verification failed: " + str(relative))
    return manifest


def finalize(run):
    supplied = Path(run).absolute()
    if supplied.is_symlink():
        raise ValueError("Refusing to delete a symlinked run directory")
    run = supplied.resolve()
    config = read(run / "config.json")
    if not config.get("cleanup_scratch_on_success", False):
        return dict(status="disabled", scratch_retained=True)
    if not terminal_success(run, config):
        return dict(status="not_complete", scratch_retained=True)
    owner = read(run / OWNER)
    root = Path(config["scratch_root"]).resolve()
    archive = Path(config["archive_directory"]).resolve()
    expected = dict(
        token=config["cleanup_owner_token"],
        run=str(run),
        scratch_root=str(root),
        archive=str(archive),
    )
    if (
        owner != expected
        or run.parent != root
        or inside(archive, root)
        or inside(run, archive)
    ):
        raise ValueError("Scratch cleanup ownership/boundary check failed")
    files = source_files(run)
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        required = sum(path.stat().st_size for path in files)
        if shutil.disk_usage(archive.parent).free < required + 5 * 1024**3:
            raise OSError(
                "Insufficient archive space with 5 GiB headroom; retaining scratch"
            )
        staging = archive.with_name(archive.name + ".partial-" + owner["token"])
        if staging.exists():
            if staging.is_symlink() or read(staging / OWNER) != owner:
                raise ValueError("Unrecognized partial archive; retaining scratch")
            shutil.rmtree(staging)
        staging.mkdir()
        save(staging / OWNER, owner)
        records = []
        for source in files:
            relative = source.relative_to(run)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            before = source.stat()
            shutil.copy2(source, target)
            source_hash = digest(source)
            if (
                digest(target) != source_hash
                or source.stat().st_mtime_ns != before.st_mtime_ns
                or source.stat().st_size != before.st_size
            ):
                raise ValueError("Result changed during archive copy: " + str(relative))
            records.append(
                dict(path=str(relative), bytes=before.st_size, sha256=source_hash)
            )
        save(
            staging / MANIFEST,
            dict(
                owner=owner,
                files=records,
                created_unix=time.time(),
                cache_included=False,
            ),
        )
        verify_archive(staging, owner)
        staging.rename(archive)
    manifest = verify_archive(archive, owner)
    saved = {row["path"]: row for row in manifest["files"]}
    if set(saved) != {str(p.relative_to(run)) for p in files}:
        raise ValueError("Scratch results differ from archive; retaining scratch")
    for source in files:
        if digest(source) != saved[str(source.relative_to(run))]["sha256"]:
            raise ValueError("Scratch file changed since export; retaining scratch")
    # Repeat the ownership check immediately before deletion. The marker and
    # exact direct-child boundary prevent a broad scratch-directory cleanup.
    if read(run / OWNER) != owner or run.resolve().parent != root:
        raise ValueError("Scratch ownership changed; retaining scratch")
    shutil.rmtree(run)
    receipt = dict(
        status="archived_and_cleaned",
        archive=str(archive),
        scratch=str(run),
        files=len(saved),
        bytes=sum(row["bytes"] for row in manifest["files"]),
        cache_deleted=True,
        completed_unix=time.time(),
    )
    save(archive / "cleanup_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    print(json.dumps(finalize(parser.parse_args().run), indent=2))
