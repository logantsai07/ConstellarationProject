import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

POST="/tmp/claude-0/-home-claude/22f51c76-c7d8-564b-9905-15120c27db37/scratchpad/poster"
CSV=f"{POST}/all_results_with_descriptors.csv"
BUDGET,TOL,NFP=3,0.01,3.0
df=pd.read_csv(CSV,low_memory=False)
PROB={"C_sparse_dipole":"f_l1_Phi","D_thin_dipole":"f_max_Phi","F_low_max_force_dipole":"f_max_force_cyl"}
LAB={"C_sparse_dipole":"sparse dipole","D_thin_dipole":"thin dipole","F_low_max_force_dipole":"min-force dipole"}
MIRROR="metrics.edge_magnetic_mirror_ratio"; IOTA="metrics.edge_rotational_transform_over_n_field_periods"
def num(s): return pd.to_numeric(s,errors="coerce")
def rank(a): return pd.Series(np.asarray(a,float)).rank().to_numpy()
def pear(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3: return float("nan")
    x,y=x[m],y[m];xc,yc=x-x.mean(),y-y.mean();d=np.sqrt((xc**2).sum()*(yc**2).sum())
    return float((xc*yc).sum()/d) if d>0 else float("nan")
def spear(x,y): return pear(rank(x),rank(y))
def feas(p):
    o=PROB[p]; s=df[(df["problem"]==p)&(df["target_index"]==BUDGET)].copy()
    for c in ("f_B","fB_target",o): s[c]=num(s[c])
    s=s[np.isfinite(s["f_B"])&np.isfinite(s["fB_target"])]
    return s[s["f_B"]<=(1+TOL)*s["fB_target"]]
import seaborn as sns
sns.set_theme(style="darkgrid", context="talk", font="DejaVu Sans")
def row(col,scale,xlabel,fname,color):
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,p in zip(axes,PROB):
        s=feas(p); cost=num(s[PROB[p]]).to_numpy(); x=np.abs(num(s[col]).to_numpy())*scale
        ok=np.isfinite(x)&np.isfinite(cost)&(cost>0)
        ax.scatter(x[ok],cost[ok],s=32,alpha=0.6,edgecolor="white",linewidth=0.3,color=color)
        if ok.sum()>2: ax.set_yscale("log")
        ax.set_title(f"{LAB[p]}   $\\rho$={spear(x[ok],cost[ok]):+.2f}",fontsize=18)
        ax.set_xlabel(xlabel,fontsize=17); ax.set_ylabel("coil cost",fontsize=16)
    fig.tight_layout()
    fig.savefig(f"{POST}/figs/{fname}",dpi=200,bbox_inches="tight",facecolor="white"); plt.close(fig)
    print("wrote",fname)
row(MIRROR,1.0,"edge mirror ratio","mirror_row.png","#2c7fb8")
row(IOTA,NFP,"|edge iota|","iota_row.png","#b0632a")
from PIL import Image
for f in ["mirror_row.png","iota_row.png"]:
    w,h=Image.open(f"{POST}/figs/{f}").size; print(f,w,h,round(w/h,3))
