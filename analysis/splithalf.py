import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings('ignore')
D=pd.read_csv('merged_features.csv')
Z=[c for c in D if c.endswith('_z') and c!='aspect_hw_z']
rng=np.random.default_rng(1)
def resid(d,o):
    cov='C(site)'+('' if o=='age' else ' + age')+(' + sex')
    return smf.ols(f'{o} ~ {cov}',d).fit().resid
res={}
for o in ['k6_log','edu','age','social','health']:
    d=D[list(dict.fromkeys([o,'age','sex','site']+Z))].dropna().reset_index(drop=True)
    if o=='age': d['y']=smf.ols('age ~ C(site)+sex',d).fit().resid
    else: d['y']=resid(d,o)
    rs=[]
    for it in range(300):
        idx=rng.permutation(len(d)); a,b=idx[:len(d)//2],idx[len(d)//2:]
        da,db=d.iloc[a],d.iloc[b]
        r=np.array([np.corrcoef(da[z],da.y)[0,1] for z in Z])
        sel=np.abs(r)>0.1
        if sel.sum()==0: rs.append(0); continue
        comp_b=(db[Z].values[:,sel]*np.sign(r[sel])).mean(1)
        rs.append(np.corrcoef(comp_b,db.y)[0,1])
    rs=np.array(rs); res[o]=(rs.mean(), np.percentile(rs,[2.5,97.5]), (rs>0).mean())
    print(o, 'held-out r mean %.3f  95%%range %s  P(r>0)=%.2f'%(rs.mean(), np.round(np.percentile(rs,[2.5,97.5]),3), (rs>0).mean()))
