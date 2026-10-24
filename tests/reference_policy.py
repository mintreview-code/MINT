"""Frozen LF policy excerpts used only for parity tests (117-residue fixture)."""

from core import *

import random

def distance(a,b):return sum(x!=y for x,y in zip(a,b))

def sign(direction):
    if direction not in ('deimmunize','immunize'):raise ValueError('Unknown direction')
    return 1 if direction=='deimmunize' else -1

def key(r,direction='deimmunize'):
    s=sign(direction)
    return (s*r['profile']['below_2'],s*r['profile']['strong_sum'],r['id'])

def unique(records,direction='deimmunize'):
    seen=set();out=[]
    for r in sorted(records,key=lambda r:key(r,direction)):
        if r['sequence'] not in seen:out.append(r);seen.add(r['sequence'])
    return out

def select(records,width,diverse_slots=0,score_slack=None,direction='deimmunize'):
    """Keep the best, then reserve slots for Hamming diversity within a score band."""
    remaining=unique(records,direction)
    if not remaining or width<=0:return []
    chosen=[];s=sign(direction);best=s*remaining[0]['profile']['below_2']
    exploit=max(1,width-diverse_slots)
    for r in remaining:
        if all(distance(r['sequence'],x['sequence'])>=2 for x in chosen):chosen.append(r)
        if len(chosen)>=exploit:break
    if not chosen:chosen=remaining[:1]
    while len(chosen)<min(width,len(remaining)):
        options=[r for r in remaining if r['sequence'] not in {x['sequence'] for x in chosen}]
        if score_slack is not None:options=[r for r in options if s*r['profile']['below_2']<=best+score_slack]
        if not options:break
        pick=max(options,key=lambda r:(min(distance(r['sequence'],x['sequence']) for x in chosen),-s*r['profile']['below_2'],-s*r['profile']['strong_sum'])) if diverse_slots else options[0]
        chosen.append(pick)
    return chosen

def proposals(parent,profile,mpnn,config,round_number):
    direction=config['direction'];rng=random.Random(config['seed']+round_number*100003+int(sha(parent['sequence'])[:8],16))
    proposal_profile=dict(profile)
    if direction=='immunize':
        # Immunization must also create binders where no <2% hit exists yet.
        current={h['position']:h['coverage'] for h in profile['hotspots']}
        proposal_profile['hotspots']=[dict(position=i,coverage=current.get(i,0)) for i in sorted(set(range(1,118))-m.FIXED,key=lambda i:(-current.get(i,0),i))]
    candidates=m.propose_singles(proposal_profile,max_positions=config['max_positions'])
    for c in candidates:c['kind']='single'
    # Proactive paired substitutions; do not wait for a single-mutation plateau.
    pairs=profile['pairs'][:config['position_pairs']]
    if direction=='immunize' and len(pairs)<config['position_pairs']:
        sites=[h['position'] for h in proposal_profile['hotspots'][:16]]
        pairs=[dict(positions=list(p)) for p in itertools.combinations(sites,2)][:config['position_pairs']]
    alternatives={}
    for pos in sorted({i for pair in pairs for i in pair['positions']}):
        _,prob=mpnn.sample(parent,[pos],1.,rng.randrange(2**31))
        allowed=[(a,prob[pos-1][j]) for j,a in enumerate(m.AA) if a not in ('C',parent['sequence'][pos-1]) and prob[pos-1][j]>=.01]
        alternatives[pos]=sorted(allowed,key=lambda x:(-x[1],x[0]))[:3]
    for pair in pairs:
        i,j=pair['positions']
        for (a,pa),(b,pb) in itertools.product(alternatives[i],alternatives[j]):
            seq=list(parent['sequence']);seq[i-1]=a;seq[j-1]=b
            candidates.append(dict(sequence=''.join(seq),kind='pair',mpnn_marginal_product=pa*pb))
    sites=[h['position'] for h in proposal_profile['hotspots'][:32]]
    for index in range(config['joint_mpnn_per_parent']):
        if not sites:break
        positions=rng.sample(sites,min(len(sites),(2,4,8)[index%3]));seed=rng.randrange(2**31)
        seq,_=mpnn.sample(parent,positions,config['temperatures'][index%3],seed)
        candidates.append(dict(sequence=seq,kind='joint_mpnn',mpnn_seed=seed,mpnn_positions=positions))
    result=[];seen={parent['sequence']}
    for c in candidates:
        if c['sequence'] in seen:continue
        seen.add(c['sequence']);geometry.invariant(c['sequence'],config['root_sequence'])
        c.update(id='r%d_%s_%s'%(round_number,sha(parent['sequence'])[:10],sha(c['sequence'])[:12]),
                 parent_id=parent['id'],founder=parent.get('founder',parent['id']))
        result.append(c)
    return result

class ReferenceMapper(m.Mapper):
    def profile(self,sequence):
        p=super().profile(sequence);coverage=Counter();pairs=Counter()
        for r in p['rows']:
            if r['rank']<2:
                sites=[i for i in range(r['start'],r['end']+1) if i not in self.fixed]
                coverage.update(sites);pairs.update(itertools.combinations(sites,2))
        p['hotspots']=[dict(position=i,coverage=v) for i,v in sorted(coverage.items(),key=lambda x:(-x[1],x[0]))]
        p['pairs']=[dict(positions=list(i),coverage=v) for i,v in sorted(pairs.items(),key=lambda x:(-x[1],x[0]))]
        return p

    def screen2(self,parent,candidates,allowance,direction='deimmunize'):
        """Only <2% is constrained; strict bands are measured, not separate stages."""
        if parent['alleles']!=self.alleles or parent['scorer_fingerprint']!=self.cache.backend.fingerprint:raise ValueError('Parent provenance mismatch')
        if allowance<0:raise ValueError('Negative allowance')
        s=sign(direction)
        if len({c['id'] for c in candidates})!=len(candidates):raise ValueError('Duplicate ids')
        work=[];old=parent['sequence'];windows=m.windows(old)
        for c in candidates:
            seq=c['sequence'];m.windows(seq)
            if len(seq)!=len(old):raise ValueError('Length changed')
            changed={i+1 for i,(a,b) in enumerate(zip(old,seq)) if a!=b}
            if changed&self.fixed:raise ValueError('Fixed residue changed')
            affected=[(s,n,p) for s,n,p in windows if any(s<=i<s+n for i in changed)]
            keys={(s,n) for s,n,p in affected}
            removed=m.counts([r['rank'] for r in parent['rows'] if (r['start'],r['length']) in keys])
            work.append(dict(candidate=c,peptides=[seq[s-1:s-1+n] for s,n,p in affected],removed=removed,
                             partial=[0,0,0],nalleles=0,rejected=False,changed_positions=sorted(changed)))
        order=sorted(self.alleles,key=lambda a:(-parent['per_allele'][a][2],a))
        for offset in range(0,len(order),5):
            active=[w for w in work if not w['rejected']]
            if not active:break
            panel=order[offset:offset+5];ranks=self.cache.get([p for w in active for p in w['peptides']],panel)
            for w in active:
                part=m.counts([ranks[a,p] for a in panel for p in w['peptides']])
                w['partial']=[a+b for a,b in zip(w['partial'],part)];w['nalleles']+=len(panel)
                if s==1:w['rejected']=w['partial'][2]>w['removed'][2]+allowance
                else:
                    maximum=w['partial'][2]+len(w['peptides'])*(len(order)-w['nalleles'])
                    w['rejected']=maximum<w['removed'][2]-allowance
        out=[]
        for w in work:
            r=dict(w['candidate'],complete_counts=w['nalleles']==len(order),eligible=not w['rejected'],
                   alleles_processed=w['nalleles'],partial_affected_counts_lower_bound=w['partial'],changed_positions=w['changed_positions'])
            if r['eligible'] and not r['complete_counts']:raise AssertionError('Incomplete acceptance')
            if r['complete_counts']:
                counts=[a-b+c for a,b,c in zip(parent['counts'],w['removed'],w['partial'])]
                r['profile']=dict(counts=counts,strong_sum=sum(counts[:2]),below_2=counts[2])
            out.append(r)
        return out
