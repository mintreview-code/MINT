"""Small real MPNN + NetMHCpan test; no GPU/folds and no job submission."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *
from models import MPNN

def main(run):
    config=read(run/'config.json');config['root_pdb']=str(run/config['root_pdb'])
    m.configure(config);geometry.configure(config);temp=Path(os.environ['TMPDIR'])
    parent=dict(id='root',sequence=config['root_sequence'],pdb=Path(config['root_pdb']).read_text())
    start=cpu();model=MPNN(config,temp)
    free=sorted(set(range(1,len(parent['sequence'])+1))-m.FIXED)
    single,prob=model.sample(parent,free[:1],1.,87101)
    joint,_=model.sample(parent,free[:min(4,len(free))],.7,87102)
    fresh,_=model.sample(parent,free,.7,87103,True)
    for seq in (single,joint,fresh):geometry.invariant(seq,parent['sequence'])
    assert len(prob)==len(parent['sequence']) and len(prob[0])==20
    fast=Backend(temp/'fast',workers=1);stock=Backend(temp/'stock',workers=1,accelerated=False)
    try:
        peptides=list(dict.fromkeys([p for _,_,p in m.windows(parent['sequence'])][:2]+[p for _,_,p in m.windows(joint)][:2]))
        a=fast.one((peptides,m.ALLELES));b=stock.one((peptides,m.ALLELES))
        assert a==b
        cache=Cache(temp/'smoke.sqlite',fast);assert cache.start_rows==0
        first=cache.get(peptides,m.ALLELES);before=cache.stats['predicted_pairs'];second=cache.get(peptides,m.ALLELES)
        assert first==second==a and cache.stats['predicted_pairs']==before
        cache.db.close()
        import torch,numpy,prody,Bio
        save(run/'cpu_smoke_passed.json',dict(passed=True,length=len(parent['sequence']),fixed=len(m.FIXED),
             direction=config['direction'],mpnn_single=single,mpnn_joint=joint,mpnn_fresh=fresh,
             exact_stock_fast_pairs=len(a),native_accelerated=fast.accelerated,cold_cache_and_reuse_verified=True,
             cpu_seconds=cpu()-start,torch=torch.__version__,numpy=numpy.__version__,prody=prody.__version__,biopython=Bio.__version__,
             limitation='CPU integration only; real GPU folding is not exercised by this test.'))
    finally:fast.close();stock.close()
    print('Real MPNN / exact predictor / cache integration passed')

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
