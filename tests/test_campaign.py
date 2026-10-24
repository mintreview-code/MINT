"""Round boundaries, interrupted fold reuse, temperatures and cleanup safety."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run120 as r
import search
import test_cleanup

class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)
        self.e=r.Experiment120.__new__(r.Experiment120)
        self.e.run=self.path;self.e.c={'seed':100,'allocation_search_hours':42,'rounds':120,'direction':'deimmunize'}
        self.e.finishing=False;self.e.stop=False;self.e.start_wall=r.time.monotonic();self.e.journal={}
        self.e.design_cpu=lambda:10.;self.e.elapsed=lambda:20.
        self.items=[dict(id=str(i),sequence='A'*12+str(i),pdb='PDB') for i in range(8)]
    def test_pause_raises_instead_of_silently_completing_partial_round(self):
        self.e.stop=True
        with self.assertRaises(r.Pause):self.e.expired()
    def test_allocation_budget_is_not_cumulative(self):
        self.e.initial_wall=1000*3600
        self.assertFalse(self.e.expired())
    def test_old_allocation_pauses(self):
        self.e.start_wall-=43*3600
        with self.assertRaises(r.Pause):self.e.expired()
    def test_partial_fold_reused_without_refolding(self):
        calls=[]
        def fold(items,seed):
            calls.append(seed);self.e.stop=True
            return [dict(x,passes=True) for x in items]
        self.e.af3=SimpleNamespace(fold=fold)
        with self.assertRaises(r.Pause):self.e.fold(self.items,5)
        self.assertEqual(len(r.read(self.path/'pending_folds.json.gz')['5']['records']),4)
        self.e.stop=False;self.e.journal=r.read(self.path/'pending_folds.json.gz')
        result=self.e.fold(self.items,5)
        self.assertEqual(len(result),8);self.assertEqual(calls,[50100,50104])
    def test_changed_pending_shortlist_fails_closed(self):
        self.e.journal={'1':dict(signature=['wrong'],records=[])}
        with self.assertRaises(ValueError):self.e.fold(self.items,1)
    def test_pause_writes_no_complete_marker(self):
        self.e.state={'round':7};self.e.initial_af3_wall=3
        self.e.af3=SimpleNamespace(wall_seconds=4)
        self.e.cache=SimpleNamespace(snapshot=lambda p:p.write_text('cache'))
        self.e.pause()
        self.assertEqual(r.read(self.path/'paused.json')['completed_rounds'],7)
        self.assertFalse((self.path/'complete.json').exists())
        self.assertTrue((self.path/'cache.sqlite').exists())
    def test_temperature_calls_are_separate(self):
        import ast

        tree = ast.parse((r.HERE / 'search.py').read_text())
        experiment = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'Experiment'
        )
        methods = {
            node.name: node for node in experiment.body
            if isinstance(node, ast.FunctionDef)
        }
        proposals = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == 'proposals'
        )

        def assert_expression_present(node, expression):
            # Compare Python structure so formatting cannot alter this check.
            expected = ast.dump(ast.parse(expression, mode='eval').body)
            self.assertTrue(
                any(ast.dump(child) == expected for child in ast.walk(node)),
                f'{node.name} must contain {expression}',
            )

        assert_expression_present(
            methods['initial'],
            "self.c.get('initial_temperatures', self.c['temperatures'])",
        )
        assert_expression_present(
            methods['one_round'],
            "self.c.get('immigrant_temperatures', self.c['temperatures'])",
        )
        assert_expression_present(
            proposals, "config['temperatures'][index % len(config['temperatures'])]",
        )
        for n in range(512):
            c={'initial_temperatures':[.4],'immigrant_temperatures':[.4],'temperatures':[.4,.7,1.]}
            self.assertEqual(c['initial_temperatures'][n%len(c['initial_temperatures'])],.4)
    def test_finalization_not_run_on_pause(self):
        self.e.state={'round':7};self.e.cache=SimpleNamespace(db=SimpleNamespace(close=lambda:None));self.e.backend=SimpleNamespace(close=lambda:None)
        with patch.object(self.e,'expired',side_effect=r.Pause()),patch.object(self.e,'pause') as paused,patch.object(self.e,'finish') as finish:
            self.e.main()
        paused.assert_called_once();finish.assert_not_called()

if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(ResumeTests),
                            unittest.defaultTestLoader.loadTestsFromTestCase(test_cleanup.CleanupTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
