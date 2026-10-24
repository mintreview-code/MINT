"""Portable exact NetMHCpan profiles; native settings are configured per run."""

from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
AA = "ACDEFGHIKLMNPQRSTVWY"
THRESHOLDS = (0.25, 0.5, 2.0)
ALLELES = json.loads((HERE / "assets/alleles25.json").read_text())
FIXED = set()
INSTALL = Path("/net/software/netmhcpan/netMHCpan-4.1")
NATIVE_MODE = "auto"
ENGINE = "nnalign_gaps_pan_play_MA_wgt_MN_v2_two_outputs_context_allelelist_pboth"
ENGINE_SHA = "ed7475343f5112e0b58fd6b1c3b22625a100423364ded32e15d34f3d64878305"


def configure(config):
    global ALLELES, FIXED, INSTALL, NATIVE_MODE
    ALLELES = list(config["alleles"])
    FIXED = set(config["fixed_positions"])
    INSTALL = Path(config["netmhcpan"])
    NATIVE_MODE = config.get("native_mode", "auto")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cpu():
    return sum(
        r.ru_utime + r.ru_stime
        for r in (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
    )


def windows(sequence):
    if not sequence or set(sequence) - set(AA):
        raise ValueError("Canonical amino acids required")
    return [
        (i + 1, n, sequence[i : i + n])
        for n in (9, 10, 11)
        for i in range(len(sequence) - n + 1)
    ]


def counts(ranks):
    # These counts are cumulative thresholds, so strong hits appear in each
    # looser threshold as well; they are not disjoint rank bins.
    return [sum(r < t for r in ranks) for t in THRESHOLDS]


class Runner:
    def __init__(self, installation, scratch):
        installation = Path(installation)
        self.native_root = installation / "Linux_x86_64"
        self.binary = self.native_root / "bin/netMHCpan"
        self.scratch = Path(scratch)
        self.env = dict(
            os.environ,
            NMHOME=str(installation),
            NETMHCpan=str(self.native_root),
            TMPDIR=str(scratch),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        self.env.pop("LD_PRELOAD", None)
        self.env.pop("NATIVE_SIMD_VERIFY", None)

    def call(self, peptides, alleles, label):
        with tempfile.NamedTemporaryFile(mode="w", dir=self.scratch) as f:
            f.write("\n".join(peptides) + "\n")
            f.flush()
            command = [str(self.binary), "-p", f.name, "-a", ",".join(alleles)]
            result = subprocess.run(
                command, env=self.env, capture_output=True, text=True, timeout=600
            )
        if result.returncode:
            raise RuntimeError("NetMHCpan failed: " + result.stderr[-2000:])
        rows = []
        for line in result.stdout.splitlines():
            fields = line.split()
            if fields and fields[0].isdigit() and len(fields) >= 13:
                rows.append((fields[1].replace("*", ""), fields[2], float(fields[12])))
        expected = [(a.replace("*", ""), p) for a in alleles for p in peptides]
        if [(a, p) for a, p, r in rows] != expected:
            raise ValueError("Unexpected NetMHCpan row order/count")
        return rows, result.stdout, result.stderr


class NativeBackend:
    def __init__(self, scratch, accelerated=False):
        assets = {}
        for sub in ("data", "Linux_x86_64/data", "Linux_x86_64/bin"):
            for directory, _, files in os.walk(INSTALL / sub, followlinks=True):
                for name in sorted(files):
                    p = Path(directory) / name
                    assets[str(p.relative_to(INSTALL))] = digest(p)
        engine = digest(INSTALL / "Linux_x86_64/bin" / ENGINE)
        self.metadata = dict(
            engine=engine,
            assets=assets,
            mode="peptide -p; EL %Rank",
            parser_sha=digest(__file__),
            kernel_sha=None,
            schema=1,
            context="no flanking context, canonical 9-11mers",
        )
        import sys

        self.probe = sys.modules[__name__]


class Mapper:
    def __init__(self, cache, alleles=None, fixed=None):
        self.cache = cache
        self.alleles = list(ALLELES if alleles is None else alleles)
        self.fixed = set(FIXED if fixed is None else fixed)
        if not self.alleles or len(set(self.alleles)) != len(self.alleles):
            raise ValueError("Unique nonempty allele panel required")

    def profile(self, sequence):
        before = cpu()
        ws = windows(sequence)
        ranks = self.cache.get([p for _, _, p in ws], self.alleles)
        rows = [
            dict(
                start=s, end=s + n - 1, length=n, peptide=p, allele=a, rank=ranks[a, p]
            )
            for a in self.alleles
            for s, n, p in ws
        ]
        total = counts([r["rank"] for r in rows])
        return dict(
            sequence=sequence,
            sequence_sha256=sha(sequence.encode()),
            alleles=self.alleles,
            scorer_fingerprint=self.cache.backend.fingerprint,
            complete=True,
            counts=total,
            strong_sum=sum(total[:2]),
            below_2=total[2],
            rows=rows,
            per_allele={
                a: counts([r["rank"] for r in rows if r["allele"] == a])
                for a in self.alleles
            },
            map_cpu_seconds=cpu() - before,
        )


def propose_singles(parent, max_positions=None):
    if max_positions is None:
        max_positions = len(parent["sequence"]) - len(FIXED)
    if not 0 <= max_positions <= len(parent["sequence"]) - len(FIXED):
        raise ValueError("Invalid mutable-site limit")
    out = []
    for h in parent["hotspots"][:max_positions]:
        i = h["position"]
        if i in FIXED:
            raise ValueError("Fixed hotspot")
        for a in AA:
            if a in ("C", parent["sequence"][i - 1]):
                continue
            seq = list(parent["sequence"])
            seq[i - 1] = a
            out.append(dict(id="single_%d%s" % (i, a), sequence="".join(seq)))
    return out
