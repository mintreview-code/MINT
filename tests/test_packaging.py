"""Preparation presets and UI boundaries without models, containers, or jobs."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lf
import prepare


class PackagingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name).resolve()
        self.site = self.path / "site.json"
        self.site.write_text(
            json.dumps(
                {
                    "cpu_container": "",
                    "cpu_python": sys.executable,
                    "netmhcpan": str(self.path / "external/netmhcpan"),
                    "af3_image": str(self.path / "external/af3.sif"),
                    "af3_model_dir": str(self.path / "external/models"),
                    "cpus": 8,
                }
            )
        )
        # Preparation still parses real inputs and writes its actual config.
        # Only external installation availability is mocked.
        availability = patch.object(lf, "check_site", return_value={})
        availability.start()
        self.addCleanup(availability.stop)

    def prepare(self, name="run", **kwargs):
        run = prepare.prepare_run(self.path / name, self.site, **kwargs)
        return run, lf.read(run / "config.json")

    def assert_temperatures_and_resources(self, config):
        self.assertEqual(config["initial_temperatures"], [0.4])
        self.assertEqual(config["immigrant_temperatures"], [0.4])
        self.assertEqual(config["temperatures"], [0.4])
        self.assertEqual(config["objective_mode"], "weighted8")
        self.assertEqual(config["native_mode"], "required")
        self.assertFalse(config["cleanup_scratch_on_success"])
        self.assertEqual(config["cpus"], 8)
        self.assertEqual(config["predictor_workers"], 7)
        self.assertEqual(config["gpus"], 1)
        self.assertEqual(config["campaign_preset"], "weighted8_t04_20261018")

    def test_campaign_preset_matches_revised_settings(self):
        run, config = self.prepare()
        expected = {
            "initial_designs": 512,
            "initial_folds": 96,
            "folds_per_round": 48,
            "beam": 24,
            "rounds": 120,
            "search_hours": 42,
            "allocation_search_hours": 42,
            "seed": 26051011,
        }
        self.assertEqual({key: config[key] for key in expected}, expected)
        self.assert_temperatures_and_resources(config)
        fixed = [
            int(token[1:])
            for token in (prepare.HERE / "examples/luxsit_fixed.txt")
            .read_text()
            .split()
        ]
        self.assertEqual(config["fixed_positions"], fixed)
        self.assertEqual(len(fixed), 43)
        self.assertEqual(len(config["root_sequence"]), 117)
        self.assertTrue((run / "input.pdb").is_file())
        self.assertFalse((run / "submission.json").exists())

    def test_smoke_preset_matches_original_small_run(self):
        _, config = self.prepare(smoke=True)
        expected = {
            "initial_designs": 4,
            "initial_folds": 5,
            "folds_per_round": 2,
            "beam": 2,
            "rounds": 1,
            "search_hours": 1.5,
            "allocation_search_hours": 1.5,
            "seed": 202610071,
            "parents_per_round": 1,
            "position_pairs": 1,
            "joint_mpnn_per_parent": 2,
            "immigrants": 0,
        }
        self.assertEqual({key: config[key] for key in expected}, expected)
        self.assert_temperatures_and_resources(config)

    def test_custom_chain_does_not_inherit_luxsit_fixed_positions(self):
        lines = []
        for line in prepare.EXAMPLE.read_text().splitlines():
            if line.startswith("ATOM") and int(line[22:26]) <= 23:
                number = int(line[22:26]) + 39
                lines.append(line[:21] + "B" + f"{number:4d}" + line[26:])
        custom = self.path / "custom.pdb"
        custom.write_text("\n".join(lines) + "\nEND\n")

        run, config = self.prepare(pdb=custom, chain="B")
        self.assertEqual(len(config["root_sequence"]), 23)
        self.assertEqual(config["fixed_positions"], [])
        self.assertEqual(
            lf.read(run / "residue_mapping.json")[0]["original_id"], "B:40"
        )

        _, explicit = self.prepare(
            "with_fixed", pdb=custom, chain="B", fixed="B:40 B:45"
        )
        self.assertEqual(explicit["fixed_positions"], [1, 6])

    def test_explicit_settings_override_bundled_example_defaults(self):
        _, config = self.prepare(
            fixed_indices="2", seed=17, native_mode="off", smoke=True
        )
        self.assertEqual(config["fixed_positions"], [2])
        self.assertEqual(config["seed"], 17)
        self.assertEqual(config["native_mode"], "off")

    def test_existing_run_is_rejected_without_overwriting(self):
        run, _ = self.prepare()
        original_config = (run / "config.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            prepare.prepare_run(run, self.site, direction="immunize")
        self.assertEqual((run / "config.json").read_bytes(), original_config)

    def test_ui_preparation_does_not_submit_a_job(self):
        import app

        out = self.path / "ui_run"
        with patch.object(lf, "submit") as submit, patch.object(
            lf.subprocess, "run"
        ) as execute:
            message, run, config = app.prepare_from_ui(
                str(self.site),
                str(out),
                None,
                "",
                "",
                "immunize",
                29,
                "off",
                False,
            )
        submit.assert_not_called()
        execute.assert_not_called()
        self.assertIn("No job submitted", message)
        self.assertEqual(Path(run), out)
        self.assertEqual(config, lf.read(out / "config.json"))
        self.assertEqual(config["direction"], "immunize")
        self.assertEqual(config["seed"], 29)
        self.assertEqual(config["native_mode"], "off")
        self.assertEqual(len(config["fixed_positions"]), 43)
        self.assertFalse((out / "submission.json").exists())

        # An empty UI seed uses the same original smoke preset as the CLI.
        with patch.object(lf, "submit") as submit:
            _, _, smoke = app.prepare_from_ui(
                str(self.site),
                str(self.path / "ui_smoke"),
                None,
                "",
                "",
                "deimmunize",
                None,
                "off",
                True,
            )
        submit.assert_not_called()
        self.assertEqual(smoke["seed"], 202610071)
        self.assertEqual(smoke["allocation_search_hours"], 1.5)


if __name__ == "__main__":
    unittest.main()
