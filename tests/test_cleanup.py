"""Cleanup must never remove unarchived scientific results or resumable runs."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import archive_results as archive

class CleanupTests(unittest.TestCase):
    def setUp(self):
        # Fixtures are tiny; model free space explicitly so the 5 GiB production
        # safety check does not make these tests depend on the host disk.
        from types import SimpleNamespace
        space = patch.object(archive.shutil, 'disk_usage', return_value=SimpleNamespace(free=10 * 1024**3))
        space.start(); self.addCleanup(space.stop)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'scratch';self.run=self.root/'lineage'
        self.run.mkdir(parents=True);self.dest=self.base/'permanent/lineage'
        self.c=dict(direction='deimmunize',rounds=120,root_sequence='A'*117,alleles=['HLA-A02:01'])
        self.c.update(archive.initialize(self.run,self.root,self.dest))
        self.write('config.json',self.c)
        self.done=dict(complete=True,direction='deimmunize',rounds=120,counts=[0,0,5],
                       original_predictor_audit=True,confirmation_passes=2,confirmation_total=2,stopped_at_budget=False)
        self.write('complete.json',self.done)
        self.write('state.json.gz',dict(round=120,archive=[{'id':'champion'}]))
        self.write('confirmation.json.gz',dict(records=[{'passes':True},{'passes':True}]))
        for name in ('champion.json','provenance.json'):self.write(name,{'retained':True})
        for name in ('champion.pdb','input.pdb','progress.csv'):(self.run/name).write_text('scientific result\n')
        (self.run/'cache.sqlite').write_bytes(b'cache contents')
        (self.run/'cache.sqlite.tmp').write_bytes(b'checkpoint replacement')
    def write(self,name,value):
        path=self.run/name
        with gzip.open(path,'wt') if path.suffix=='.gz' else path.open('w') as f:json.dump(value,f)
    def test_success_archives_science_and_deletes_scratch(self):
        outcome=archive.finalize(self.run)
        self.assertEqual(outcome['status'],'archived_and_cleaned');self.assertFalse(self.run.exists())
        self.assertTrue((self.dest/'champion.pdb').exists());self.assertTrue((self.dest/'state.json.gz').exists())
        self.assertFalse((self.dest/'cache.sqlite').exists());self.assertFalse((self.dest/'cache.sqlite.tmp').exists())
        archive.verify_archive(self.dest,archive.read(self.dest/archive.OWNER))
    def test_default_off_retains_cache(self):
        self.c['cleanup_scratch_on_success']=False;self.write('config.json',self.c)
        self.assertEqual(archive.finalize(self.run)['status'],'disabled');self.assertTrue((self.run/'cache.sqlite').exists())
    def test_pause_retains_cache(self):
        (self.run/'complete.json').unlink()
        self.assertEqual(archive.finalize(self.run)['status'],'not_complete');self.assertTrue((self.run/'cache.sqlite').exists())
    def test_budget_stop_not_completion(self):
        self.done.update(stopped_at_budget=True);self.write('complete.json',self.done)
        self.assertTrue(archive.finalize(self.run)['scratch_retained'])
    def test_failed_audit_retained(self):
        self.done.update(original_predictor_audit=False);self.write('complete.json',self.done)
        self.assertTrue(archive.finalize(self.run)['scratch_retained'])
    def test_early_stop_requires_verified_endpoint(self):
        self.done.update(rounds=50,counts=[0,0,0],confirmation_passes=1);self.write('complete.json',self.done)
        self.write('state.json.gz',{'round':50});self.write('confirmation.json.gz',{'records':[{'passes':True},{'passes':False}]})
        self.assertTrue(archive.finalize(self.run)['scratch_retained'])
    def test_verified_zero_endpoint_can_cleanup(self):
        self.done.update(rounds=50,counts=[0,0,0]);self.write('complete.json',self.done);self.write('state.json.gz',{'round':50})
        self.assertEqual(archive.finalize(self.run)['status'],'archived_and_cleaned')
    def test_weighted_immunization_requires_full_endpoint_for_early_cleanup(self):
        self.c.update(direction='immunize',objective_mode='weighted8')
        self.write('config.json',self.c)
        # Every occurrence below 2%, but only in the final cumulative bucket.
        cutoff_counts={str(t):0 for t in archive.THRESHOLDS}
        cutoff_counts['2.0']=324
        self.done.update(direction='immunize',rounds=50,counts=[0,0,324],
                         objective_mode='weighted8',objective_score=324,cutoff_counts=cutoff_counts)
        self.write('complete.json',self.done);self.write('state.json.gz',{'round':50})
        self.assertTrue(archive.finalize(self.run)['scratch_retained'])
        self.assertTrue(self.run.exists())
        # All occurrences now contribute at every threshold: 8 * 324.
        self.done.update(counts=[324,324,324],objective_score=2592,
                         cutoff_counts={str(t):324 for t in archive.THRESHOLDS})
        self.write('complete.json',self.done)
        self.assertEqual(archive.finalize(self.run)['status'],'archived_and_cleaned')
    def test_export_failure_retains_all_source(self):
        with patch.object(archive.shutil,'copy2',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):archive.finalize(self.run)
        self.assertTrue((self.run/'cache.sqlite').exists());self.assertTrue((self.run/'champion.pdb').exists())
        self.assertEqual(archive.finalize(self.run)['status'],'archived_and_cleaned')
    def test_insufficient_archive_space_retains_cache(self):
        from types import SimpleNamespace
        with patch.object(archive.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaises(OSError):archive.finalize(self.run)
        self.assertTrue((self.run/'cache.sqlite').exists())
    def test_corrupt_archive_blocks_deletion(self):
        with patch.object(archive.shutil,'rmtree',side_effect=OSError('interrupted before deletion')):
            with self.assertRaises(OSError):archive.finalize(self.run)
        (self.dest/'champion.pdb').write_text('corrupt')
        with self.assertRaises(ValueError):archive.finalize(self.run)
        self.assertTrue((self.run/'cache.sqlite').exists())
    def test_symlinks_rejected(self):
        (self.run/'external').symlink_to(self.base)
        with self.assertRaises(ValueError):archive.finalize(self.run)
        self.assertTrue(self.run.exists())
    def test_owner_tampering_rejected(self):
        owner=archive.read(self.run/archive.OWNER);owner['token']='wrong';self.write(archive.OWNER,owner)
        with self.assertRaises(ValueError):archive.finalize(self.run)
        self.assertTrue(self.run.exists())
    def test_archive_inside_scratch_rejected(self):
        with self.assertRaises(ValueError):archive.initialize(self.run,self.root,self.root/'archive')
    def test_missing_result_blocks_cleanup(self):
        (self.run/'champion.pdb').unlink()
        with self.assertRaises(ValueError):archive.finalize(self.run)
        self.assertTrue((self.run/'cache.sqlite').exists())

if __name__=='__main__':unittest.main()
