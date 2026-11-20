import argparse,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def dataset(n=2400,seed=27):
 r=np.random.default_rng(seed);x=r.normal(size=(n,7));prop=1/(1+np.exp(-(.7*x[:,0]-.4*x[:,1])));t=r.binomial(1,prop);tau=.5+1.2*(x[:,2]>0)-.7*x[:,3];base=.4*x[:,0]+.6*x[:,1]-.3*x[:,4];y=base+t*tau+r.normal(scale=.8,size=n);return x,t,y,tau,base
def evaluate(seed=27):
 x,t,y,tau,base=dataset(seed=seed);cut=1600;m0=RandomForestRegressor(n_estimators=120,max_depth=7,random_state=seed).fit(x[:cut][t[:cut]==0],y[:cut][t[:cut]==0]);m1=RandomForestRegressor(n_estimators=120,max_depth=7,random_state=seed+1).fit(x[:cut][t[:cut]==1],y[:cut][t[:cut]==1]);effect=m1.predict(x[cut:])-m0.predict(x[cut:]);true=tau[cut:];policy=effect>0;oracle=true>0;return {"pehe":float(np.sqrt(np.mean((effect-true)**2))),"policy_agreement":float(np.mean(policy==oracle)),"estimated_policy_value":float(np.mean(base[cut:]+policy*true)),"treat_all_value":float(np.mean(base[cut:]+true))}
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
