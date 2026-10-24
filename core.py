"""Exact rank cache, cumulative-rank objective, and population selection."""

import concurrent.futures
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
import threading
import time
from collections import Counter, defaultdict

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


import mapper as m
import geometry
import objective as scoring

objective = scoring.score

cpu = m.cpu
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()


def read(path):
    path = Path(path)
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as f:
        return json.load(f)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with (
        gzip.open(tmp, "wt", compresslevel=3) if path.suffix == ".gz" else tmp.open("w")
    ) as f:
        json.dump(value, f, allow_nan=False, separators=(",", ":"))
        f.write("\n")
    tmp.replace(path)


def metric(profile):
    result = {
        k: v for k, v in profile.items() if k not in ("rows", "hotspots", "pairs")
    }
    if "rows" in profile:
        result["cutoff_counts"] = {
            str(t): sum(r["rank"] < t for r in profile["rows"])
            for t in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
        }
    return scoring.annotate(result)


def distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def sign(direction):
    if direction not in ("deimmunize", "immunize"):
        raise ValueError("Unknown direction")
    return 1 if direction == "deimmunize" else -1


def key(r, direction="deimmunize"):
    # Both directions use the same ordering: the sign reverses the objectives.
    s = sign(direction)
    return (s * objective(r["profile"]), s * r["profile"]["strong_sum"], r["id"])


def unique(records, direction="deimmunize"):
    seen = set()
    out = []
    for r in sorted(records, key=lambda r: key(r, direction)):
        if r["sequence"] not in seen:
            out.append(r)
            seen.add(r["sequence"])
    return out


def select(records, width, diverse_slots=0, score_slack=None, direction="deimmunize"):
    """Keep the best, then reserve slots for Hamming diversity within a score band."""
    remaining = unique(records, direction)
    if not remaining or width <= 0:
        return []
    chosen = []
    s = sign(direction)
    best = s * objective(remaining[0]["profile"])
    exploit = max(1, width - diverse_slots)
    for r in remaining:
        if all(distance(r["sequence"], x["sequence"]) >= 2 for x in chosen):
            chosen.append(r)
        if len(chosen) >= exploit:
            break
    if not chosen:
        chosen = remaining[:1]
    while len(chosen) < min(width, len(remaining)):
        options = [
            r for r in remaining if r["sequence"] not in {x["sequence"] for x in chosen}
        ]
        if score_slack is not None:
            options = [
                r for r in options if s * objective(r["profile"]) <= best + score_slack
            ]
        if not options:
            break
        pick = (
            max(
                options,
                key=lambda r: (
                    min(distance(r["sequence"], x["sequence"]) for x in chosen),
                    -s * objective(r["profile"]),
                    -s * r["profile"]["strong_sum"],
                ),
            )
            if diverse_slots
            else options[0]
        )
        chosen.append(pick)
    return chosen


def diversity(records):
    ds = [
        distance(a["sequence"], b["sequence"])
        for a, b in itertools.combinations(records, 2)
    ]
    return dict(
        n=len(records),
        mean_distance=sum(ds) / len(ds) if ds else 0,
        min_distance=min(ds) if ds else 0,
        max_distance=max(ds) if ds else 0,
        variable_positions=[
            i + 1
            for i in range(len(records[0]["sequence"]) if records else 0)
            if len({r["sequence"][i] for r in records}) > 1
        ],
    )


class Backend:
    """Predictor threads carry NO private rank caches; inputs are disjoint missing keys."""

    def __init__(self, scratch, workers=7, accelerated=True):
        self.scratch = Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        # False avoids NativeBackend's optional rebuild of the old kernel.
        audit = m.NativeBackend(self.scratch, accelerated=False)
        self.metadata = dict(audit.metadata)
        self.kernel = HERE / "native/kernel.so"
        can_accelerate = (
            self.metadata["engine"] == m.ENGINE_SHA
            and "avx2" in Path("/proc/cpuinfo").read_text()
            and self.kernel.exists()
        )
        if accelerated and m.NATIVE_MODE == "required" and not can_accelerate:
            raise RuntimeError(
                "Verified native acceleration unavailable for this engine/CPU"
            )
        accelerated = accelerated and m.NATIVE_MODE != "off" and can_accelerate
        self.metadata["kernel_sha"] = m.digest(self.kernel) if accelerated else None
        self.fingerprint = m.sha(json.dumps(self.metadata, sort_keys=True).encode())
        self.accelerated = accelerated
        self.probe = audit.probe
        self.probe.emit = lambda *a, **k: None
        self.local = threading.local()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)

    def one(self, task):
        peptides, alleles = task
        if not hasattr(self.local, "runner"):
            path = self.scratch / str(threading.get_ident())
            path.mkdir(exist_ok=True)
            self.local.runner = self.probe.Runner(m.INSTALL, path)
            self.local.runner.env.pop("LD_PRELOAD", None)
            self.local.runner.env.pop("NATIVE_SIMD_VERIFY", None)
            if self.accelerated:
                self.local.runner.env["LD_PRELOAD"] = str(self.kernel)
        rows, _, stderr = self.local.runner.call(
            peptides, alleles, "fresh_missing_ranks"
        )
        if self.accelerated and "NATIVE_SIMD" not in stderr:
            raise RuntimeError("Acceleration hook missing")
        return {(a, p): rank for a, p, rank in rows}

    def many(self, tasks):
        return self.pool.map(self.one, tasks)

    def close(self):
        self.pool.shutdown(wait=True)


class Cache:
    """One SQLite owner on local disk; gather all misses before parallel dispatch."""

    def __init__(self, path, backend):
        self.path = Path(path).resolve()
        if Path(os.environ["TMPDIR"]).resolve() not in self.path.parents:
            raise ValueError("Cache must be job-local")
        self.backend = backend
        self.stats = Counter()
        self.db = sqlite3.connect(self.path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS ranks(namespace TEXT,peptide TEXT,allele TEXT,rank REAL NOT NULL,PRIMARY KEY(namespace,peptide,allele))"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS provenance(namespace TEXT PRIMARY KEY,metadata TEXT)"
        )
        namespaces = {r[0] for r in self.db.execute("SELECT namespace FROM provenance")}
        if namespaces and namespaces != {backend.fingerprint}:
            raise ValueError("Cache provenance changed")
        self.db.execute(
            "INSERT OR IGNORE INTO provenance VALUES (?,?)",
            (backend.fingerprint, json.dumps(backend.metadata, sort_keys=True)),
        )
        self.db.commit()
        self.start_rows = self.db.execute("SELECT count(*) FROM ranks").fetchone()[0]

    def get(self, peptides, alleles):
        peptides = list(dict.fromkeys(peptides))
        alleles = list(dict.fromkeys(alleles))
        panel = set(alleles)
        if not panel <= set(m.ALLELES):
            raise ValueError("Unknown allele")
        if any(not 9 <= len(p) <= 11 or set(p) - set(m.AA) for p in peptides):
            raise ValueError("Invalid peptide")
        result = {}
        ns = self.backend.fingerprint
        for offset in range(0, len(peptides), 256):
            block = peptides[offset : offset + 256]
            query = (
                "SELECT allele,peptide,rank FROM ranks WHERE namespace=? AND peptide IN ("
                + ",".join("?" for _ in block)
                + ")"
            )
            result.update(
                ((a, p), r)
                for a, p, r in self.db.execute(query, [ns] + block)
                if a in panel
            )
        self.stats["requested_pairs"] += len(peptides) * len(alleles)
        self.stats["hits"] += len(result)
        missing = defaultdict(list)
        for p in peptides:
            wanted = tuple(a for a in alleles if (a, p) not in result)
            if wanted:
                missing[wanted].append(p)
        tasks = []
        for wanted, group in missing.items():
            for i in range(0, len(group), 512):
                for j in range(0, len(wanted), 5):
                    tasks.append((group[i : i + 512], list(wanted[j : j + 5])))
        # Keys are disjoint across tasks. Validate each reply before insertion.
        for task, answer in zip(tasks, self.backend.many(tasks)):
            expected = {(a, p) for a in task[1] for p in task[0]}
            if set(answer) != expected or any(
                not math.isfinite(r) or not 0 <= r <= 100 for r in answer.values()
            ):
                raise ValueError("Invalid predictor response")
            with self.db:
                self.db.executemany(
                    "INSERT INTO ranks VALUES (?,?,?,?)",
                    [(ns, p, a, r) for (a, p), r in answer.items()],
                )
            result.update(answer)
            self.stats["predicted_pairs"] += len(answer)
            self.stats["calls"] += 1
        return result

    def snapshot(self, destination):
        self.db.commit()
        destination = Path(destination)
        tmp = destination.with_name(destination.name + ".tmp")
        shutil.copyfile(self.path, tmp)
        tmp.replace(destination)


class Mapper(m.Mapper):
    def profile(self, sequence):
        p = super().profile(sequence)
        coverage = Counter()
        pairs = Counter()
        for r in p["rows"]:
            if r["rank"] < 2:
                sites = [
                    i for i in range(r["start"], r["end"] + 1) if i not in self.fixed
                ]
                coverage.update(sites)
                pairs.update(itertools.combinations(sites, 2))
        scoring.annotate(p)
        p["hotspots"] = [
            dict(position=i, coverage=v)
            for i, v in sorted(coverage.items(), key=lambda x: (-x[1], x[0]))
        ]
        p["pairs"] = [
            dict(positions=list(i), coverage=v)
            for i, v in sorted(pairs.items(), key=lambda x: (-x[1], x[0]))
        ]
        return p

    def screen2(self, parent, candidates, allowance, direction="deimmunize"):
        """Exact selected-objective delta bounds; no independent hard band constraints."""
        if (
            parent["alleles"] != self.alleles
            or parent["scorer_fingerprint"] != self.cache.backend.fingerprint
        ):
            raise ValueError("Parent provenance mismatch")
        if allowance < 0:
            raise ValueError("Negative allowance")
        s = sign(direction)
        if len({c["id"] for c in candidates}) != len(candidates):
            raise ValueError("Duplicate ids")
        work = []
        old = parent["sequence"]
        windows = m.windows(old)
        for c in candidates:
            seq = c["sequence"]
            m.windows(seq)
            if len(seq) != len(old):
                raise ValueError("Length changed")
            changed = {i + 1 for i, (a, b) in enumerate(zip(old, seq)) if a != b}
            if changed & self.fixed:
                raise ValueError("Fixed residue changed")
            affected = [
                (s, n, p)
                for s, n, p in windows
                if any((s <= i < s + n for i in changed))
            ]
            keys = {(s, n) for s, n, p in affected}
            removed = scoring.cumulative(
                (r["rank"] for r in parent["rows"] if (r["start"], r["length"]) in keys)
            )
            work.append(
                dict(
                    candidate=c,
                    peptides=[seq[s - 1 : s - 1 + n] for s, n, p in affected],
                    removed=removed,
                    partial=[0] * 8,
                    nalleles=0,
                    rejected=False,
                    changed_positions=sorted(changed),
                )
            )
        # Prioritize alleles by their contribution to the selected objective.
        # Keep every window occurrence, including repeated peptide strings.
        parent_ranks = {a: [] for a in self.alleles}
        for row in parent["rows"]:
            parent_ranks[row["allele"]].append(row["rank"])
        allele_scores = {
            a: scoring.vector_score(scoring.cumulative(ranks))
            for a, ranks in parent_ranks.items()
        }
        order = sorted(self.alleles, key=lambda a: (-allele_scores[a], a))
        for offset in range(0, len(order), 5):
            active = [w for w in work if not w["rejected"]]
            if not active:
                break
            panel = order[offset : offset + 5]
            ranks = self.cache.get([p for w in active for p in w["peptides"]], panel)
            for w in active:
                part = scoring.cumulative(
                    (ranks[a, p] for a in panel for p in w["peptides"])
                )
                w["partial"] = [a + b for a, b in zip(w["partial"], part)]
                w["nalleles"] += len(panel)
                if s == 1:
                    w["rejected"] = (
                        scoring.vector_score(w["partial"])
                        > scoring.vector_score(w["removed"]) + allowance
                    )
                else:
                    maximum = scoring.vector_score(w["partial"]) + len(
                        w["peptides"]
                    ) * (len(order) - w["nalleles"]) * (
                        8 if scoring.MODE == "weighted8" else 1
                    )
                    w["rejected"] = (
                        maximum < scoring.vector_score(w["removed"]) - allowance
                    )
        out = []
        for w in work:
            r = dict(
                w["candidate"],
                complete_counts=w["nalleles"] == len(order),
                eligible=not w["rejected"],
                alleles_processed=w["nalleles"],
                partial_affected_counts_lower_bound=[
                    w["partial"][i] for i in (0, 1, 7)
                ],
                changed_positions=w["changed_positions"],
                partial_affected_cutoff_counts_lower_bound=dict(
                    zip(map(str, scoring.THRESHOLDS), w["partial"])
                ),
                partial_affected_objective_lower_bound=scoring.vector_score(
                    w["partial"]
                ),
                objective_mode=scoring.MODE,
            )
            if r["eligible"] and (not r["complete_counts"]):
                raise AssertionError("Incomplete acceptance")
            if r["complete_counts"]:
                all_counts = [
                    a - b + c
                    for a, b, c in zip(
                        scoring.vector(parent), w["removed"], w["partial"]
                    )
                ]
                counts = [all_counts[i] for i in (0, 1, 7)]
                r["profile"] = scoring.annotate(
                    dict(
                        counts=counts,
                        strong_sum=sum(counts[:2]),
                        below_2=counts[2],
                        cutoff_counts=dict(
                            zip(map(str, scoring.THRESHOLDS), all_counts)
                        ),
                    )
                )
            out.append(r)
        return out
