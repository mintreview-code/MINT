#!/usr/bin/env python3
"""Matched one-GPU searches: direction changes the selected objective, not the budget."""
from core import *
from models import MPNN, AF3
import argparse
import csv
import random
import signal


def proposals(parent, profile, mpnn, config, round_number):
    direction = config["direction"]
    rng = random.Random(
        config["seed"] + round_number * 100003 + int(sha(parent["sequence"])[:8], 16)
    )
    proposal_profile = dict(profile)
    if direction == "immunize":
        # Immunization must also create binders where no <2% hit exists yet.
        current = {h["position"]: h["coverage"] for h in profile["hotspots"]}
        proposal_profile["hotspots"] = [
            dict(position=i, coverage=current.get(i, 0))
            for i in sorted(
                set(range(1, len(config["root_sequence"]) + 1)) - m.FIXED,
                key=lambda i: (-current.get(i, 0), i),
            )
        ]
    candidates = m.propose_singles(
        proposal_profile, max_positions=config["max_positions"]
    )
    for c in candidates:
        c["kind"] = "single"
    # Proactive paired substitutions; do not wait for a single-mutation plateau.
    pairs = profile["pairs"][: config["position_pairs"]]
    if direction == "immunize" and len(pairs) < config["position_pairs"]:
        sites = [h["position"] for h in proposal_profile["hotspots"][:16]]
        pairs = [dict(positions=list(p)) for p in itertools.combinations(sites, 2)][
            : config["position_pairs"]
        ]
    alternatives = {}
    for pos in sorted({i for pair in pairs for i in pair["positions"]}):
        _, prob = mpnn.sample(parent, [pos], 1.0, rng.randrange(2**31))
        allowed = [
            (a, prob[pos - 1][j])
            for j, a in enumerate(m.AA)
            if a not in ("C", parent["sequence"][pos - 1]) and prob[pos - 1][j] >= 0.01
        ]
        alternatives[pos] = sorted(allowed, key=lambda x: (-x[1], x[0]))[:3]
    for pair in pairs:
        i, j = pair["positions"]
        for (a, pa), (b, pb) in itertools.product(alternatives[i], alternatives[j]):
            seq = list(parent["sequence"])
            seq[i - 1] = a
            seq[j - 1] = b
            candidates.append(
                dict(sequence="".join(seq), kind="pair", mpnn_marginal_product=pa * pb)
            )
    sites = [h["position"] for h in proposal_profile["hotspots"][:32]]
    for index in range(config["joint_mpnn_per_parent"]):
        if not sites:
            break
        positions = rng.sample(sites, min(len(sites), (2, 4, 8)[index % 3]))
        seed = rng.randrange(2**31)
        seq, _ = mpnn.sample(
            parent,
            positions,
            config["temperatures"][index % len(config["temperatures"])],
            seed,
        )
        candidates.append(
            dict(
                sequence=seq,
                kind="joint_mpnn",
                mpnn_seed=seed,
                mpnn_positions=positions,
            )
        )
    result = []
    seen = {parent["sequence"]}
    for c in candidates:
        if c["sequence"] in seen:
            continue
        seen.add(c["sequence"])
        geometry.invariant(c["sequence"], config["root_sequence"])
        c.update(
            id="r%d_%s_%s"
            % (round_number, sha(parent["sequence"])[:10], sha(c["sequence"])[:12]),
            parent_id=parent["id"],
            founder=parent.get("founder", parent["id"]),
        )
        result.append(c)
    return result


class Experiment:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.config = read(self.run / "config.json")
        self.c = self.config
        self.c["root_pdb"] = str(self.run / self.c["root_pdb"])
        scoring.configure(self.c.get("objective_mode", "broad2"))
        m.configure(self.c)
        geometry.configure(self.c)
        for path, digest in self.c["protected_sources"].items():
            if m.digest(HERE / path) != digest:
                raise ValueError("Frozen source changed: " + path)
        if m.digest(self.c["root_pdb"]) != self.c["input_sha256"]:
            raise ValueError("Normalized input PDB changed")
        self.temp = Path(os.environ["TMPDIR"])
        self.start_cpu = cpu()
        self.start_wall = time.monotonic()
        self.excluded_cpu = 0.0
        self.stop = False
        self.state = (
            read(self.run / "state.json.gz")
            if (self.run / "state.json.gz").exists()
            else None
        )
        self.initial_cpu = self.state["design_cpu_seconds"] if self.state else 0.0
        self.initial_wall = self.state["elapsed_seconds"] if self.state else 0.0
        signal.signal(signal.SIGUSR1, lambda *_: setattr(self, "stop", True))
        self.backend = Backend(
            self.temp / "predictor", workers=self.c["predictor_workers"]
        )
        cachefile = self.temp / "ranks.sqlite"
        if self.state or (self.run / "resume_accounting.json").exists():
            if not (self.run / "cache.sqlite").exists():
                raise ValueError("Missing checkpoint cache")
            shutil.copyfile(self.run / "cache.sqlite", cachefile)
        elif cachefile.exists() or (self.run / "cache.sqlite").exists():
            raise ValueError("New arm must start with an empty cache")
        self.cache = Cache(cachefile, self.backend)
        if (
            not self.state
            and not (self.run / "resume_accounting.json").exists()
            and self.cache.start_rows != 0
        ):
            raise ValueError("Preloaded cache forbidden")
        self.mapper = Mapper(self.cache, self.c["alleles"])
        self.mpnn = MPNN(self.c, self.temp)
        self.af3 = AF3(self.c, self.temp)
        self.initial_af3_wall = (
            self.state.get("af3_wall_seconds", 0.0) if self.state else 0.0
        )
        save(
            self.run / "provenance.json",
            dict(
                metadata=self.backend.metadata,
                fingerprint=self.backend.fingerprint,
                imported_prediction_cache=False,
                resumed_own_cache=bool(
                    self.state or (self.run / "resume_accounting.json").exists()
                ),
                starting_rows=self.cache.start_rows,
            ),
        )

    def elapsed(self):
        return self.initial_wall + time.monotonic() - self.start_wall

    def design_cpu(self):
        return (
            self.initial_cpu
            + cpu()
            - self.start_cpu
            - self.af3.driver_cpu_seconds
            - self.excluded_cpu
        )

    def expired(self):
        return self.stop or self.elapsed() >= self.c["search_hours"] * 3600

    def fold(self, items, round_number):
        records = []
        for first in range(0, len(items), 4):
            if self.expired():
                break
            block = items[first : first + 4]
            records.extend(
                self.af3.fold(block, self.c["seed"] + round_number * 10000 + first)
            )
            save(
                self.run / "live.json",
                dict(
                    phase="fold",
                    round=round_number,
                    completed=len(records),
                    planned=len(items),
                    design_cpu_seconds=self.design_cpu(),
                    elapsed_seconds=self.elapsed(),
                ),
            )
        return records

    def checkpoint(self, state):
        if state.get("archive"):
            best = min(state["archive"], key=lambda r: key(r, self.c["direction"]))
            state["history"][-1].update(
                cutoff_counts=best["profile"]["cutoff_counts"],
                objective_mode=scoring.MODE,
                objective_score=objective(best["profile"]),
                weighted_sum=best["profile"]["weighted_sum"],
            )
        self.cache.snapshot(self.run / "cache.sqlite")
        state.update(
            design_cpu_seconds=self.design_cpu(),
            elapsed_seconds=self.elapsed(),
            af3_wall_seconds=self.initial_af3_wall + self.af3.wall_seconds,
            cache_stats=dict(self.cache.stats),
        )
        save(self.run / "state.json.gz", state)
        self.state = state

    def initial(self):
        # The starting PDB seeds fresh MPNN draws. Only sequences that pass the
        # AF3 gate below can become parents in the first search population.
        root = dict(
            id="root",
            sequence=self.c["root_sequence"],
            pdb=Path(self.c["root_pdb"]).read_text(),
            founder="root",
        )
        root["profile"] = metric(self.mapper.profile(root["sequence"]))
        save(self.run / "root.json", {k: v for k, v in root.items() if k != "pdb"})
        records = [root]
        seen = {root["sequence"]}
        free = sorted(set(range(1, len(self.c["root_sequence"]) + 1)) - m.FIXED)
        for index in range(self.c["initial_designs"]):
            if self.expired():
                break
            temp = self.c.get("initial_temperatures", self.c["temperatures"])[
                index % len(self.c.get("initial_temperatures", self.c["temperatures"]))
            ]
            seed = self.c["seed"] + index
            sequence, _ = self.mpnn.sample(root, free, temp, seed, from_scratch=True)
            if sequence in seen:
                continue
            seen.add(sequence)
            identifier = "seed_" + sha(sequence)[:16]
            records.append(
                dict(
                    id=identifier,
                    founder=identifier,
                    sequence=sequence,
                    pdb=root["pdb"],
                    mpnn_seed=seed,
                    temperature=temp,
                )
            )
        # Deduplicate missing ranks across EACH initial batch, not just each sequence.
        for first in range(0, len(records), 32):
            block = records[first : first + 32]
            self.cache.get(
                [p for r in block for _, _, p in m.windows(r["sequence"])],
                self.c["alleles"],
            )
            for r in block:
                r["profile"] = metric(self.mapper.profile(r["sequence"]))
            save(
                self.run / "live.json",
                dict(
                    phase="initial_scoring",
                    completed=min(first + 32, len(records)),
                    planned=len(records),
                    design_cpu_seconds=self.design_cpu(),
                ),
            )
        shortlist = select(
            records,
            self.c["initial_folds"],
            self.c["initial_folds"] // 4,
            direction=self.c["direction"],
        )
        if root["sequence"] not in {r["sequence"] for r in shortlist}:
            shortlist.append(root)
        folded = self.fold(shortlist, 0)
        passing = [r for r in folded if r["passes"]]
        # A score-selected shortlist may contain no confident folds. Try the
        # remaining fresh draws before declaring initialization unsuccessful.
        if not passing and not self.expired():
            attempted = {r["sequence"] for r in folded}
            remaining = select(
                [r for r in records if r["sequence"] not in attempted],
                len(records),
                direction=self.c["direction"],
            )
            for first in range(0, len(remaining), 16):
                if self.expired():
                    break
                extra = self.fold(remaining[first : first + 16], 1000 + first // 16)
                folded.extend(extra)
                passing.extend(r for r in extra if r["passes"])
                if passing:
                    break
        save(
            self.run / "initial.json.gz",
            dict(
                designs=[{k: v for k, v in r.items() if k != "pdb"} for r in records],
                folds=folded,
            ),
        )
        if not passing:
            raise RuntimeError(
                "No initial design passed AF3; no search parents available"
            )
        archive = select(passing, self.c["beam"], direction=self.c["direction"])
        population = select(
            passing, self.c["beam"], self.c["beam"] // 3, direction=self.c["direction"]
        )
        best = min(archive, key=lambda r: key(r, self.c["direction"]))
        self.checkpoint(
            dict(
                round=0,
                archive=archive,
                population=population,
                history=[
                    dict(
                        round=0,
                        counts=best["profile"]["counts"],
                        design_cpu_seconds=self.design_cpu(),
                        folded=len(folded),
                        passing=len(passing),
                        diversity=diversity(population),
                    )
                ],
            )
        )

    def one_round(self, n):
        state = self.state
        direction = self.c["direction"]
        s = sign(direction)
        population = unique(state["population"] + state["archive"], direction)
        rotate = ((n - 1) * self.c["parents_per_round"]) % len(population)
        rotating = population[rotate:] + population[:rotate]
        parents = [population[0]] + [
            r for r in rotating if r["sequence"] != population[0]["sequence"]
        ][: self.c["parents_per_round"] - 1]
        candidates = []
        fold_pool = []
        seen_folded = set()
        for history in self.run.glob("round_*/results.json.gz"):
            old = read(history)
            seen_folded.update(r["sequence"] for r in old["folds"])
        seen_folded.update(r["sequence"] for r in population)
        for parent in parents:
            if self.expired():
                break
            full = self.mapper.profile(parent["sequence"])
            items = proposals(parent, full, self.mpnn, self.c, n)
            allowance = max(
                2, math.ceil(self.c["worsening_fraction"] * objective(full))
            )
            result = self.mapper.screen2(full, items, allowance, direction)
            for r in result:
                r["parent_below_2"] = full["below_2"]
                r["parent_objective"] = objective(full)
                r["allowance"] = allowance
                if r["eligible"] and r["sequence"] not in seen_folded:
                    r["exploratory"] = (
                        s * (objective(r["profile"]) - objective(full)) > 0
                    )
                    fold_pool.append(dict(r, pdb=parent["pdb"]))
            candidates.extend(result)
            save(
                self.run / "live.json",
                dict(
                    phase="screening",
                    round=n,
                    parents_done=len({r["parent_id"] for r in candidates}),
                    parents_planned=len(parents),
                    candidates=len(candidates),
                    design_cpu_seconds=self.design_cpu(),
                    cache_stats=dict(self.cache.stats),
                ),
            )
        # Periodic global immigrants originate from the ORIGINAL structure.
        if n % self.c["immigrant_interval"] == 0 and not self.expired():
            root = dict(
                id="root",
                sequence=self.c["root_sequence"],
                pdb=Path(self.c["root_pdb"]).read_text(),
                founder="root",
            )
            for i in range(self.c["immigrants"]):
                sequence, _ = self.mpnn.sample(
                    root,
                    sorted(set(range(1, len(self.c["root_sequence"]) + 1)) - m.FIXED),
                    self.c.get("immigrant_temperatures", self.c["temperatures"])[
                        i
                        % len(
                            self.c.get("immigrant_temperatures", self.c["temperatures"])
                        )
                    ],
                    self.c["seed"] + n * 100000 + i,
                    True,
                )
                identifier = "imm_%d_%s" % (n, sha(sequence)[:12])
                r = dict(
                    id=identifier,
                    founder=identifier,
                    parent_id="root",
                    sequence=sequence,
                    profile=metric(self.mapper.profile(sequence)),
                    kind="immigrant",
                    exploratory=True,
                    eligible=True,
                    complete_counts=True,
                )
                candidates.append(r)
                if sequence not in seen_folded:
                    fold_pool.append(dict(r, pdb=root["pdb"]))
        width = self.c["folds_per_round"]
        strict = [r for r in fold_pool if not r["exploratory"]]
        selected = select(strict, 3 * width // 4, 0, direction=direction)
        selected_seqs = {r["sequence"] for r in selected}
        extras = select(
            [r for r in fold_pool if r["sequence"] not in selected_seqs],
            width - len(selected),
            width - len(selected) - 1,
            direction=direction,
        )
        selected += extras
        # Audit the full sequence/allele map before AF3, even for paired proposals.
        for r in selected:
            exact = self.mapper.profile(r["sequence"])
            if exact["counts"] != r["profile"]["counts"] or scoring.vector(
                exact
            ) != scoring.vector(r["profile"]):
                raise ValueError("Eight-threshold delta/full scan mismatch")
            r["profile"] = metric(exact)
        folded = self.fold(selected, n)
        passed = [r for r in folded if r["passes"]]
        archive = select(state["archive"] + passed, self.c["beam"], direction=direction)
        before = min(state["archive"], key=lambda r: key(r, direction))
        champion = min(archive, key=lambda r: key(r, direction))
        assert s * objective(champion["profile"]) <= s * objective(before["profile"])
        slack = max(10, math.ceil(0.1 * objective(champion["profile"])))
        # The archive preserves the best scores; the separate population also
        # reserves space for sequence diversity inside this score allowance.
        population = select(
            state["population"] + archive + passed,
            self.c["beam"],
            self.c["beam"] // 3,
            slack,
            direction,
        )
        out = self.run / ("round_%03d" % n)
        save(out / "results.json.gz", dict(candidates=candidates, folds=folded))
        row = dict(
            round=n,
            counts=champion["profile"]["counts"],
            design_cpu_seconds=self.design_cpu(),
            candidates=len(candidates),
            folded=len(folded),
            passing=len(passed),
            plddt=champion["plddt"],
            rmsd=champion["root_ca_rmsd"],
            diversity=diversity(population),
        )
        self.checkpoint(
            dict(
                round=n,
                archive=archive,
                population=population,
                history=state["history"] + [row],
            )
        )
        save(out / "summary.json", row)

    def finish(self):
        finish_cpu = cpu()
        prior_excluded = self.excluded_cpu
        prior_af3_cpu = self.af3.driver_cpu_seconds
        direction = self.c["direction"]
        best = min(self.state["archive"], key=lambda r: key(r, direction))
        # Independent original predictor audit; no old rank cache and no acceleration.
        before = cpu()
        original = Backend(self.temp / "original_audit", workers=1, accelerated=False)
        try:
            peptides = list(dict.fromkeys(p for _, _, p in m.windows(best["sequence"])))
            ranks = original.one((peptides, self.c["alleles"]))
            counts = m.counts(
                [
                    ranks[a, p]
                    for a in self.c["alleles"]
                    for _, _, p in m.windows(best["sequence"])
                ]
            )
            if counts != best["profile"]["counts"]:
                raise ValueError("Original predictor audit mismatch")
            all_counts = scoring.cumulative(
                ranks[a, p]
                for a in self.c["alleles"]
                for _, _, p in m.windows(best["sequence"])
            )
            if all_counts != scoring.vector(best["profile"]):
                raise ValueError("Eight-threshold original predictor audit mismatch")
            save(
                self.run / "objective_audit.json",
                dict(
                    passed=True,
                    cutoff_counts=dict(zip(map(str, scoring.THRESHOLDS), all_counts)),
                    objective_mode=scoring.MODE,
                    objective_score=scoring.vector_score(all_counts),
                ),
            )
        finally:
            original.close()
            self.excluded_cpu += cpu() - before
        # Two independent single-sample AF3 repeats. Record all outcomes; do not
        # silently replace the search champion if reproducibility is marginal.
        confirmation = []
        for replicate in range(2):
            confirmation += self.af3.fold([best], self.c["seed"] + 2000000 + replicate)
        save(
            self.run / "confirmation.json.gz",
            dict(
                records=[
                    {k: v for k, v in r.items() if k != "pdb"} for r in confirmation
                ],
                all_pass=all(r["passes"] for r in confirmation),
            ),
        )
        (self.run / "champion.pdb").write_text(best["pdb"])
        save(self.run / "champion.json", {k: v for k, v in best.items() if k != "pdb"})
        with (self.run / "archive.fasta").open("w") as f:
            for r in self.state["archive"]:
                f.write(
                    ">%s counts=%s\n%s\n"
                    % (
                        r["id"],
                        ",".join(map(str, r["profile"]["counts"])),
                        r["sequence"],
                    )
                )
        history = self.state["history"]
        with (self.run / "progress.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "round",
                    "design_cpu_hours",
                    "n_lt025",
                    "n_lt05",
                    "n_lt2",
                    "folded",
                    "passing",
                    "objective_mode",
                    "objective_score",
                    "weighted_sum",
                ]
                + [f"cutoff_lt_{t:g}" for t in scoring.THRESHOLDS]
            )
            for r in history:
                writer.writerow(
                    [
                        r["round"],
                        r["design_cpu_seconds"] / 3600,
                        *r["counts"],
                        r["folded"],
                        r["passing"],
                        scoring.MODE,
                        r["objective_score"],
                        r["weighted_sum"],
                    ]
                    + [r["cutoff_counts"][str(t)] for t in scoring.THRESHOLDS]
                )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for index, name in [
            (2, "below_2"),
            (None, "strong_sum"),
            (None, "objective_score"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(
                [r["design_cpu_seconds"] / 3600 for r in history],
                [
                    (
                        r["objective_score"]
                        if name == "objective_score"
                        else (
                            r["counts"][index]
                            if index is not None
                            else sum(r["counts"][:2])
                        )
                    )
                    for r in history
                ],
                "-o",
            )
            ax.set(
                xlabel="Consumed design CPU hours (AF3/final audit excluded)",
                ylabel=name,
                title=direction,
            )
            fig.tight_layout()
            fig.savefig(self.run / (name + ".png"), dpi=160)
            plt.close(fig)
        self.excluded_cpu = (
            prior_excluded
            + cpu()
            - finish_cpu
            - (self.af3.driver_cpu_seconds - prior_af3_cpu)
        )
        self.checkpoint(self.state)
        save(
            self.run / "complete.json",
            dict(
                complete=True,
                direction=direction,
                rounds=self.state["round"],
                counts=counts,
                objective_mode=scoring.MODE,
                objective_score=objective(best["profile"]),
                cutoff_counts=best["profile"]["cutoff_counts"],
                weighted_sum=best["profile"]["weighted_sum"],
                original_predictor_audit=True,
                confirmation_passes=sum(r["passes"] for r in confirmation),
                confirmation_total=2,
                design_cpu_seconds=self.design_cpu(),
                elapsed_seconds=self.elapsed(),
                af3_wall_seconds=self.initial_af3_wall + self.af3.wall_seconds,
                stopped_at_budget=self.expired(),
                max_rounds=self.c["rounds"],
            ),
        )

    def main(self):
        try:
            if not self.state:
                self.initial()
            for n in range(self.state["round"] + 1, self.c["rounds"] + 1):
                if self.expired():
                    break
                target = objective(
                    min(
                        self.state["archive"], key=lambda r: key(r, self.c["direction"])
                    )["profile"]
                )
                if target == (
                    0
                    if self.c["direction"] == "deimmunize"
                    else len(m.windows(self.c["root_sequence"]))
                    * len(self.c["alleles"])
                    * (8 if scoring.MODE == "weighted8" else 1)
                ):
                    break
                self.one_round(n)
            self.finish()
        except Exception as exc:
            save(
                self.run / "failure.json",
                dict(
                    error=repr(exc),
                    design_cpu_seconds=self.design_cpu(),
                    elapsed_seconds=self.elapsed(),
                ),
            )
            raise
        finally:
            self.cache.db.close()
            self.backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    Experiment(parser.parse_args().run).main()
