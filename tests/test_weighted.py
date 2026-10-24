"""Weighted scoring, exact pruning, fixed-temperature draws and endpoint regressions."""

import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import core
import search
import run120
from test_portable import FakeMPNN

s = core.scoring
m = core.m


class RankCache:
    backend = types.SimpleNamespace(fingerprint="weighted-test")

    def get(self, peptides, alleles):
        values = [
            0.0,
            0.249,
            0.25,
            0.499,
            0.5,
            0.749,
            0.75,
            0.999,
            1.0,
            1.249,
            1.25,
            1.499,
            1.5,
            1.749,
            1.75,
            1.999,
            2.0,
            2.5,
            90.0,
        ]
        return {
            (a, p): values[int(core.sha(a + p)[:10], 16) % len(values)]
            for a in alleles
            for p in peptides
        }


class WeightedTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(s.configure, "broad2")
        s.configure("weighted8")
        text, seq, mapping = core.geometry.normalize(
            core.HERE / "examples/luxsit_seed.pdb"
        )
        self.parent = dict(id="root", sequence=seq, pdb=text)
        self.config = dict(
            root_sequence=seq,
            fixed_positions=[1, 6],
            alleles=core.read(core.HERE / "assets/alleles25.json"),
            netmhcpan="unused",
            direction="deimmunize",
            seed=42,
            max_positions=0,
            position_pairs=0,
            joint_mpnn_per_parent=8,
            temperatures=[0.4],
        )
        m.configure(self.config)
        core.geometry.configure(self.config)
        self.mapper = core.Mapper(RankCache(), self.config["alleles"])

    def test_strict_threshold_boundaries_and_missing_data(self):
        for rank, expected in [
            (0, 8),
            (0.249, 8),
            (0.25, 7),
            (0.5, 6),
            (0.75, 5),
            (1.0, 4),
            (1.25, 3),
            (1.5, 2),
            (1.75, 1),
            (1.999, 1),
            (2.0, 0),
        ]:
            self.assertEqual(sum(s.cumulative([rank])), expected)
        with self.assertRaisesRegex(ValueError, "All eight thresholds"):
            s.score(dict(below_2=4, counts=[1, 2, 4]))

    def test_delta_pruning_matches_full_scan_both_directions_and_modes(self):
        rng = random.Random(20261018)
        candidates = []
        for i in range(160):
            seq = list(self.parent["sequence"])
            sites = [p for p in range(1, len(seq) + 1) if p not in m.FIXED]
            for pos in rng.sample(sites, (1, 2, 4, 8)[i % 4]):
                seq[pos - 1] = rng.choice(
                    [a for a in m.AA if a not in ("C", seq[pos - 1])]
                )
            candidates.append(dict(id=str(i), sequence="".join(seq)))
        for mode in ("weighted8", "broad2"):
            s.configure(mode)
            parent = self.mapper.profile(self.parent["sequence"])
            full = {r["id"]: self.mapper.profile(r["sequence"]) for r in candidates}
            for direction in ("deimmunize", "immunize"):
                for allowance in (0, 2, 20):
                    rows = self.mapper.screen2(parent, candidates, allowance, direction)
                    for row in rows:
                        exact = full[row["id"]]
                        expected = (
                            core.sign(direction) * (s.score(exact) - s.score(parent))
                            <= allowance
                        )
                        self.assertEqual(row["eligible"], expected)
                        if row["eligible"]:
                            self.assertTrue(row["complete_counts"])
                        if row["complete_counts"]:
                            self.assertEqual(s.vector(row["profile"]), s.vector(exact))
                            self.assertEqual(row["profile"]["counts"], exact["counts"])

    def test_duplicate_windows_count_each_occurrence(self):
        seq = "A" * 30
        profile = self.mapper.profile(seq)
        ranks = RankCache().get([p for _, _, p in m.windows(seq)], m.ALLELES)
        direct = sum(
            sum(ranks[a, p] < t for t in s.THRESHOLDS)
            for a in m.ALLELES
            for _, _, p in m.windows(seq)
        )
        self.assertEqual(s.score(profile), direct)

    def test_allele_order_uses_selected_objective_and_stable_ties(self):
        class OrderedCache:
            backend = types.SimpleNamespace(fingerprint="allele-order-test")

            def __init__(self):
                self.panels = []

            def get(self, peptides, alleles):
                self.panels.append(list(alleles))
                return {
                    (a, p): (
                        1.8 if a == "Aweak" else
                        0.2 if a in ("Bstrong", "Cstrong") and len(p) == 11 else
                        2.0
                    )
                    for a in alleles for p in peptides
                }

        alleles = ["Gzero", "Fzero", "Ezero", "Dzero", "Cstrong", "Bstrong", "Aweak"]
        cache = OrderedCache()
        mapper = core.Mapper(cache, alleles, fixed=[])
        candidate = dict(id="child", sequence="AAAAAWAAAAA")
        for mode, expected in (
            ("weighted8", ["Bstrong", "Cstrong", "Aweak", "Dzero", "Ezero", "Fzero", "Gzero"]),
            ("broad2", ["Aweak", "Bstrong", "Cstrong", "Dzero", "Ezero", "Fzero", "Gzero"]),
        ):
            s.configure(mode)
            parent = mapper.profile("A" * 11)
            exact = mapper.profile(candidate["sequence"])
            # Six weak occurrences outweigh one strong occurrence only in broad2.
            self.assertEqual(parent["per_allele"]["Aweak"][2], 6)
            self.assertEqual(parent["per_allele"]["Bstrong"][2], 1)
            for direction in ("deimmunize", "immunize"):
                with self.subTest(mode=mode, direction=direction):
                    cache.panels.clear()
                    result = mapper.screen2(parent, [candidate], 0, direction)[0]
                    self.assertEqual(cache.panels, [expected[:5], expected[5:]])
                    self.assertTrue(result["eligible"])
                    self.assertTrue(result["complete_counts"])
                    self.assertEqual(s.vector(result["profile"]), s.vector(exact))

    def test_weighted_ranking_can_reverse_broad_count_ranking(self):
        def record(identifier, sequence, ranks):
            counts = m.counts(ranks)
            profile = s.annotate(
                dict(
                    rows=[dict(rank=r) for r in ranks],
                    counts=counts,
                    below_2=counts[2],
                    strong_sum=sum(counts[:2]),
                )
            )
            return dict(id=identifier, sequence=sequence, profile=profile)

        strong = record("strong", "AAAA", [0.2])  # broad=1, weighted=8
        weak = record("weak", "WWWW", [1.8, 1.8])  # broad=2, weighted=2
        self.assertEqual(core.select([strong, weak], 1)[0]["id"], "weak")
        self.assertEqual(
            core.select([strong, weak], 1, direction="immunize")[0]["id"], "strong"
        )
        self.assertEqual(len(core.select([strong, weak], 2, 1, 5)), 1)
        self.assertEqual(len(core.select([strong, weak], 2, 1, 6)), 2)

    def test_eight_joint_calls_all_use_point_four(self):
        profile = self.mapper.profile(self.parent["sequence"])
        for direction in ("deimmunize", "immunize"):
            calls = []
            fake = FakeMPNN()

            def sample(parent, positions, temperature, seed):
                calls.append((len(positions), temperature))
                return fake.sample(parent, positions, temperature, seed)

            search.proposals(
                self.parent,
                profile,
                types.SimpleNamespace(sample=sample),
                dict(self.config, direction=direction),
                1,
            )
            self.assertEqual(
                calls,
                [
                    (2, 0.4),
                    (4, 0.4),
                    (8, 0.4),
                    (2, 0.4),
                    (4, 0.4),
                    (8, 0.4),
                    (2, 0.4),
                    (4, 0.4),
                ],
            )

    def test_weighted_endpoint_requires_full_weighted_maximum(self):
        for direction, rank, should_complete in [
            ("immunize", 1.8, False),
            ("immunize", 0.2, True),
            ("deimmunize", 2.0, True),
        ]:
            with tempfile.TemporaryDirectory() as temp:
                e = run120.Experiment120.__new__(run120.Experiment120)
                e.run = Path(temp)
                e.c = dict(
                    direction=direction,
                    rounds=10,
                    root_sequence="A" * 11,
                    alleles=["test"],
                )
                ranks = [rank] * len(m.windows(e.c["root_sequence"]))
                counts = m.counts(ranks)
                profile = s.annotate(
                    dict(
                        rows=[dict(rank=r) for r in ranks],
                        below_2=counts[2],
                        strong_sum=sum(counts[:2]),
                    )
                )
                e.state = dict(
                    round=0,
                    archive=[dict(id="best", sequence="A" * 11, profile=profile)],
                )
                e.cache = types.SimpleNamespace(
                    db=types.SimpleNamespace(close=lambda: None)
                )
                e.backend = types.SimpleNamespace(close=lambda: None)
                with patch.object(e, "expired", return_value=False), patch.object(
                    e, "complete", return_value=dict(confirmation_passes=2)
                ) as complete, patch.object(
                    e, "one_round", side_effect=run120.Pause()
                ) as next_round, patch.object(
                    e, "pause"
                ):
                    e.main()
                self.assertEqual(complete.called, should_complete)
                self.assertEqual(next_round.called, not should_complete)

    def test_successful_confirmation_exports_weighted_progress(self):
        from unittest.mock import MagicMock
        import sys
        import csv

        with tempfile.TemporaryDirectory() as temp:
            e = search.Experiment.__new__(search.Experiment)
            e.run = e.temp = Path(temp)
            e.c = dict(direction="deimmunize", alleles=["test"], seed=42, rounds=1)
            e.excluded_cpu = 0
            e.initial_af3_wall = 0
            e.design_cpu = lambda: 10.0
            e.elapsed = lambda: 20.0
            e.expired = lambda: False
            sequence = "A" * 11
            ranks = [1.1] * len(m.windows(sequence))
            counts = m.counts(ranks)
            profile = s.annotate(
                dict(
                    rows=[dict(rank=r) for r in ranks],
                    counts=counts,
                    below_2=counts[2],
                    strong_sum=sum(counts[:2]),
                )
            )
            best = dict(id="best", sequence=sequence, pdb="fixture", profile=profile)
            e.state = dict(
                round=1,
                archive=[best],
                history=[
                    dict(
                        round=1,
                        counts=counts,
                        design_cpu_seconds=10.0,
                        folded=1,
                        passing=1,
                    )
                ],
            )
            e.cache = types.SimpleNamespace(snapshot=lambda p: None, stats={})
            e.af3 = types.SimpleNamespace(
                driver_cpu_seconds=0.0,
                wall_seconds=0.0,
                fold=MagicMock(return_value=[dict(passes=True)]),
            )
            e.checkpoint(e.state)
            predictor = types.SimpleNamespace(
                one=lambda task: {(a, p): 1.1 for a in task[1] for p in task[0]},
                close=lambda: None,
            )
            pyplot = MagicMock()
            pyplot.subplots.return_value = (MagicMock(), MagicMock())
            with patch.object(search, "Backend", return_value=predictor), patch.dict(
                sys.modules, {"matplotlib": MagicMock(pyplot=pyplot), "matplotlib.pyplot": pyplot}
            ):
                e.finish()
            done = core.read(e.run / "complete.json")
            self.assertEqual(done["objective_score"], 24)
            self.assertEqual(done["confirmation_passes"], 2)
            self.assertEqual(e.af3.fold.call_count, 2)
            self.assertTrue(core.read(e.run / "objective_audit.json")["passed"])
            with (e.run / "progress.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["objective_score"], "24")
            self.assertEqual(row["cutoff_lt_1.25"], "6")

    def test_final_audit_detects_difference_hidden_by_three_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            e = search.Experiment.__new__(search.Experiment)
            e.run = e.temp = Path(temp)
            e.c = dict(direction="deimmunize", alleles=["test"])
            e.excluded_cpu = 0
            e.af3 = types.SimpleNamespace(driver_cpu_seconds=0.0)
            sequence = "A" * 11
            ranks = [1.1] * len(m.windows(sequence))
            counts = m.counts(ranks)
            profile = s.annotate(
                dict(
                    rows=[dict(rank=r) for r in ranks],
                    counts=counts,
                    below_2=counts[2],
                    strong_sum=sum(counts[:2]),
                )
            )
            e.state = dict(
                archive=[dict(id="best", sequence=sequence, profile=profile)]
            )
            predictor = types.SimpleNamespace(
                one=lambda task: {(a, p): 1.8 for a in task[1] for p in task[0]},
                close=lambda: None,
            )
            with patch.object(search, "Backend", return_value=predictor):
                with self.assertRaisesRegex(
                    ValueError, "Eight-threshold original predictor audit"
                ):
                    e.finish()


if __name__ == "__main__":
    unittest.main()
