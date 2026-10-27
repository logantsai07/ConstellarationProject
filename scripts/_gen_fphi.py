import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

POST="/tmp/claude-0/-home-claude/22f51c76-c7d8-564b-9905-15120c27db37/scratchpad/poster"
CSV=f"{POST}/all_results_with_descriptors.csv"
BUDGET,TOL=3,0.01
df=pd.read_csv(CSV,low_memory=False)
PROB={"C_sparse_dipole":"f_l1_Phi","D_thin_dipole":"f_max_Phi"}
def num(s): return pd.to_numeric(s,errors="coerce")
def rank(a): return pd.Series(np.asarray(a,float)).rank().to_numpy()
def pear(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3: return float("nan")
    x,y=x[m],y[m];xc,yc=x-x.mean(),y-y.mean();d=np.sqrt((xc**2).sum()*(yc**2).sum())
    return float((xc*yc).sum()/d) if d>0 else float("nan")
def spear(x,y): return pear(rank(x),rank(y))
def feas(p,obj):
    s=df[(df["problem"]==p)&(df["target_index"]==BUDGET)].copy()
    for c in ("f_B","fB_target",obj): s[c]=num(s[c])
    s=s[np.isfinite(s["f_B"])&np.isfinite(s["fB_target"])]
    return s[s["f_B"]<=(1+TOL)*s["fB_target"]]
g=feas("G_l2_dipole","f_Phi"); G=pd.DataFrame({"config_id":g["config_id"].values,"fphi":num(g["f_Phi"]).values})
import seaborn as sns
sns.set_theme(style="darkgrid", context="talk", font="DejaVu Sans")
fig,axes=plt.subplots(1,2,figsize=(12.5,5.3))
for ax,(cp,cobj,tag) in zip(axes,[("C_sparse_dipole","f_l1_Phi","sparse  ($f_{\\ell 1}$)"),("D_thin_dipole","f_max_Phi","thin  ($f_{\\max}$)")]):
    c=feas(cp,cobj); C=pd.DataFrame({"config_id":c["config_id"].values,"cost":num(c[cobj]).values})
    m=G.merge(C,on="config_id").dropna(); x=m["fphi"].to_numpy(); y=m["cost"].to_numpy()
    ok=(x>0)&(y>0)&np.isfinite(x)&np.isfinite(y)
    ax.scatter(x[ok],y[ok],s=40,alpha=0.65,edgecolor="white",linewidth=0.3,color="#6a51a3")
    if ok.sum()>2: ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"$f_\\Phi$ vs {tag}    $\\rho$={spear(x[ok],y[ok]):+.2f}  ($n$={ok.sum()})",fontsize=16)
    ax.set_xlabel("$f_\\Phi$ achieved  (cheap L2 solve)",fontsize=15)
    ax.set_ylabel(f"{tag.split()[0]} achieved",fontsize=15)
fig.tight_layout()
fig.savefig(f"{POST}/figs/fphi_correlation.png",dpi=200,bbox_inches="tight",facecolor="white")
print("wrote fphi_correlation.png")
print("=== dims ===")
for f in sorted(os.listdir(f"{POST}/figs")):
    w,h=Image.open(f"{POST}/figs/{f}").size; print(f"{f}: {w}x{h}  ar={w/h:.3f}")
