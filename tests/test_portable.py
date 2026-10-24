"""Portable input mapping, LF-policy parity, exact scoring and AF3 protocol tests."""
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *
import tempfile
import unittest
from unittest.mock import patch
import random
from search import proposals
from models import AF3
import lf

class Fake:
    fingerprint='fake';metadata={}
    def __init__(self):self.keys=[]
    def rank(self,a,p):return [0.,.249,.25,.499,.5,1.99,2.,4.][int(sha(a+p)[:8],16)%8]
    def many(self,tasks):
        for peptides,alleles in tasks:
            self.keys.extend((a,p) for a in alleles for p in peptides)
            yield {(a,p):self.rank(a,p) for a in alleles for p in peptides}

class FakeMPNN:
    def sample(self,parent,positions,temperature,seed,from_scratch=False):
        rng=random.Random(seed);seq=list(parent['sequence'])
        for i in positions:seq[i-1]=rng.choice(m.AA.replace('C',''))
        return ''.join(seq),[[.05]*20 for _ in seq]

class PortableTests(unittest.TestCase):
    def setUp(self):
        scoring.configure("broad2");self.addCleanup(scoring.configure,"broad2")
        self.temp=tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'));self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        self.env=patch.dict(os.environ,{'TMPDIR':str(self.path)});self.env.start();self.addCleanup(self.env.stop)
        self.text,self.seq,self.mapping=geometry.normalize(HERE/'examples/luxsit_seed.pdb')
        self.fixed=geometry.fixed_positions((HERE/'examples/luxsit_fixed.txt').read_text(),self.mapping)
        self.config=dict(root_sequence=self.seq,fixed_positions=self.fixed,alleles=read(HERE/'assets/alleles25.json'),netmhcpan='/net/software/netmhcpan/netMHCpan-4.1',seed=26051011,max_positions=6,position_pairs=2,joint_mpnn_per_parent=3,temperatures=[.4,.7,1.],direction='deimmunize')
        m.configure(self.config);geometry.configure(self.config)
    def cache(self,name='cache'):
        backend=Fake();cache=Cache(self.path/(name+'.sqlite'),backend);self.addCleanup(cache.db.close);return cache,backend
    def cropped(self,length=23,chain='B',start=40):
        lines=[]
        for line in self.text.splitlines():
            if not line.startswith('ATOM'):continue
            i=int(line[22:26])
            if i<=length:lines.append(line[:21]+chain+f'{start+i-1:4d}'+line[26:])
        p=self.path/'cropped.pdb';p.write_text('\n'.join(lines)+'\nEND\n');return p
    def test_luxsit_fixed_residues_and_identity(self):
        self.assertEqual(len(self.seq),117);self.assertEqual(len(self.fixed),43)
        self.assertEqual(self.fixed,[int(x[1:]) for x in (HERE/'examples/luxsit_fixed.txt').read_text().split()])
    def test_arbitrary_length_chain_and_numbering(self):
        text,seq,mapping=geometry.normalize(self.cropped(),chain='B')
        self.assertEqual(len(seq),23);self.assertEqual(mapping[0]['original_id'],'B:40')
        self.assertEqual(geometry.fixed_positions('B:40 B45',mapping,'10-12'),[1,6,10,11,12])
        dest=self.path/'normalized.pdb';dest.write_text(text)
        self.assertEqual(geometry.pdb_data(dest)[0],seq)
    def test_multichain_requires_explicit_choice(self):
        p=self.cropped();p.write_text(p.read_text().replace('END\n','')+self.text)
        with self.assertRaises(ValueError):geometry.normalize(p)
        self.assertEqual(len(geometry.normalize(p,'B')[1]),23)
    def test_insertion_code_fixed_mapping(self):
        p=self.cropped();s=p.read_text();lines=[]
        for line in s.splitlines():
            if line.startswith('ATOM') and int(line[22:26])==46:line=line[:22]+'  45'+'A'+line[27:]
            lines.append(line)
        p.write_text('\n'.join(lines)+'\n');_,_,mapping=geometry.normalize(p)
        self.assertEqual(geometry.fixed_positions('B:45A',mapping),[7])
    def test_missing_atom_rejected(self):
        p=self.cropped();p.write_text('\n'.join(l for l in p.read_text().splitlines() if not (l.startswith('ATOM') and l[12:16].strip()=='O' and int(l[22:26])==45))+'\n')
        with self.assertRaises(ValueError):geometry.normalize(p)
    def test_chain_break_rejected(self):
        p=self.cropped();p.write_text('\n'.join(l for l in p.read_text().splitlines() if not(l.startswith('ATOM') and int(l[22:26])==45))+'\n')
        with self.assertRaises(ValueError):geometry.normalize(p)
    def test_unknown_fixed_id_rejected(self):
        with self.assertRaises(ValueError):geometry.fixed_positions('Z:999',self.mapping)
    def test_scan_sizes_and_thresholds(self):
        self.assertEqual(len(m.windows(self.seq))*25,8100)
        self.assertEqual(len(m.windows(self.seq[:23])),42)
        self.assertEqual(m.counts([.249,.25,.499,.5,1.99,2.]),[1,3,5])
    def test_cache_deduplicates_and_completes_partial_panel(self):
        c,b=self.cache();peps=[self.seq[:9],self.seq[1:10]]
        c.get(peps*3,m.ALLELES[:5]);c.get(peps,m.ALLELES)
        self.assertEqual(len(b.keys),50);c.get(peps*2,m.ALLELES);self.assertEqual(len(b.keys),50)
    def test_arbitrary_length_delta_pairs_both_directions(self):
        c,b=self.cache();config=dict(self.config,root_sequence=self.seq[:23],fixed_positions=[1,6]);m.configure(config);geometry.configure(config)
        mapper=Mapper(c);parent=mapper.profile(config['root_sequence']);seq=list(parent['sequence']);seq[8]='W' if seq[8]!='W' else 'A';seq[9]='Y' if seq[9]!='Y' else 'D'
        candidate=dict(id='pair',sequence=''.join(seq));full=mapper.profile(candidate['sequence'])
        for direction in ('deimmunize','immunize'):
            r=mapper.screen2(parent,[candidate],2,direction)[0]
            expected=full['below_2']<=parent['below_2']+2 if direction=='deimmunize' else full['below_2']>=parent['below_2']-2
            self.assertEqual(r['eligible'],expected)
            if r['complete_counts']:self.assertEqual(r['profile']['counts'],full['counts'])
    def test_fixed_mutations_rejected(self):
        c,b=self.cache();mapper=Mapper(c);p=mapper.profile(self.seq);seq=list(self.seq);seq[0]='A'
        with self.assertRaises(ValueError):mapper.screen2(p,[dict(id='bad',sequence=''.join(seq))],2)
    def test_original_lf_policy_parity_both_directions(self):
        sys.path.insert(0,str(HERE/'tests'));import reference_policy as ref
        c,b=self.cache();mapper=Mapper(c);p=mapper.profile(self.seq);parent=dict(id='root',sequence=self.seq,pdb=self.text)
        for direction in ('deimmunize','immunize'):
            cfg=dict(self.config,direction=direction)
            expected=ref.proposals(parent,p,FakeMPNN(),cfg,5)
            actual=proposals(parent,p,FakeMPNN(),cfg,5)
            self.assertEqual(actual,expected)
            reference_mapper=ref.ReferenceMapper(c)
            revised=mapper.screen2(p,actual,3,direction)
            original=reference_mapper.screen2(p,actual,3,direction)
            for got,want in zip(revised,original):
                for field,value in want.items():
                    if field=="profile":
                        self.assertEqual({k:got[field][k] for k in value},value)
                    else:self.assertEqual(got[field],value)
    def test_original_selection_policy_parity(self):
        sys.path.insert(0,str(HERE/'tests'));import reference_policy as ref
        records=[]
        for i in range(30):
            seq=list(self.seq);seq[1]=m.AA[i%20];seq[2]=m.AA[i//20]
            records.append(dict(id=str(i),sequence=''.join(seq),profile=dict(below_2=100+i%9,strong_sum=i)))
        for direction in ('deimmunize','immunize'):
            self.assertEqual(select(records,24,8,10,direction),ref.select(records,24,8,10,direction))
    def test_variable_length_af3_manifest_and_confidence(self):
        from Bio.PDB import MMCIFIO
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        text,seq,mapping=geometry.normalize(self.cropped());path=self.path/'root.pdb';path.write_text(text)
        cfg=dict(self.config,root_sequence=seq,root_pdb=str(path),fixed_positions=[1,6]);m.configure(cfg);geometry.configure(cfg)
        af3=AF3(cfg,self.path)
        def request(payload):
            items=read(payload['manifest']);self.assertEqual(len(items),1)
            for item in items:
                import io
                data=MMCIF2Dict(io.StringIO(item['cif']))
                self.assertEqual(len(data['_entity_poly_seq.num']),23)
                data['_atom_site.B_iso_or_equiv']=['95']*len(data['_atom_site.id'])
                out=Path(item['output_dir']);out.mkdir();writer=MMCIFIO();writer.set_dict(data);writer.save(str(out/(item['name']+'_model.cif')))
                save(out/(item['name']+'_summary_confidences.json'),{})
        af3.request=request
        result=af3.fold([dict(id='test',sequence=seq,pdb=text)],7)[0]
        self.assertTrue(result['passes']);self.assertEqual(result['plddt'],95);self.assertLess(result['root_ca_rmsd'],1e-5)
    def test_submission_has_no_original_account_dependency(self):
        site=lf.site_settings();self.assertNotIn('mwitwer',json.dumps(site))
        for path in (HERE/'core.py',HERE/'search.py',HERE/'models.py',HERE/'lf.py',HERE/'job.sh'):
            self.assertNotIn('/home/mwitwer',path.read_text())

if __name__=='__main__':unittest.main()
