from treatment_policy.experiment import dataset,evaluate
def test_shapes():
 x,t,y,tau,b=dataset(30,2);assert x.shape==(30,7) and all(len(v)==30 for v in [t,y,tau,b])
def test_metrics():
 r=evaluate(3);assert r["pehe"]>=0 and 0<=r["policy_agreement"]<=1

