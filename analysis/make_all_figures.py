"""make_all_figures.py
Makes all the analysis figures at the i=3 budget.

    python make_all_figures.py all_results_with_descriptors.csv figs

  iota_vs_cost.png          dipole cost vs iota
  mirror_vs_cost.png        dipole cost vs mirror ratio
  metric_colormap.png       Spearman(cost, metric) for each problem and metrics.* column
  multiple_R.png            multiple correlation R per problem (cost ~ all metrics)
  fphi_correlation.png      f_Phi vs sparse and thin
  plasma_corr_matrix.png    correlation matrix between the plasma parameters
  deconfounded_weights.png  R^-1 c, the colormap with the collinearity removed

Run collect_results.py and join_descriptors.py first. Change BUDGET for a
different f_B index.
"""

import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":13,"axes.titlesize":14,"axes.labelsize":13,
                     "xtick.labelsize":11,"ytick.labelsize":11,"figure.dpi":150})

CSV = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "figs"
os.makedirs(OUT, exist_ok=True)
BUDGET, TOL, NFP = 3, 0.01, 3.0
df = pd.read_csv(CSV, low_memory=False)

DIPOLE={"C_sparse_dipole":"f_l1_Phi","D_thin_dipole":"f_max_Phi","F_low_max_force_dipole":"f_max_force_cyl"}
PROB  ={**DIPOLE,"G_l2_dipole":"f_Phi",
        "I_low_curvature_filament":"f_max_K_dot_grad_K_cyl","K_low_max_force_filament":"f_max_force_cyl"}
SHORT ={"C_sparse_dipole":"C sparse","D_thin_dipole":"D thin","F_low_max_force_dipole":"F force",
        "G_l2_dipole":"G f_Phi","I_low_curvature_filament":"I curv","K_low_max_force_filament":"K force"}
PCOLOR={"C_sparse_dipole":"#9ecae1","D_thin_dipole":"#4292c6","F_low_max_force_dipole":"#08519c",
        "G_l2_dipole":"#6a51a3","I_low_curvature_filament":"#fb6a4a","K_low_max_force_filament":"#cb181d"}
IOTA_E="metrics.edge_rotational_transform_over_n_field_periods"
MIRROR="metrics.edge_magnetic_mirror_ratio"

def num(s): return pd.to_numeric(s, errors="coerce")
def rank(a): return pd.Series(np.asarray(a,float)).rank().to_numpy()
def pear(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3: return float("nan")
    x,y=x[m],y[m]; xc,yc=x-x.mean(),y-y.mean(); d=np.sqrt((xc**2).sum()*(yc**2).sum())
    return float((xc*yc).sum()/d) if d>0 else float("nan")
def spear(x,y): return pear(rank(x),rank(y))
def z(a):
    a=np.asarray(a,float); s=a.std(); return (a-a.mean())/s if s>0 else a*0
def feas(p, idx=BUDGET):
    o=PROB[p]
    if o not in df.columns: return df.iloc[0:0]
    s=df[(df["problem"]==p)&(df["target_index"]==idx)].copy()
    if len(s)==0: return s
    for c in ("f_B","fB_target",o): s[c]=num(s[c])
    s=s[np.isfinite(s["f_B"])&np.isfinite(s["fB_target"])]
    return s[s["f_B"]<=(1+TOL)*s["fB_target"]]

allm =[c for c in df.columns if c.startswith("metrics.") and num(df[c]).notna().sum()>10 and num(df[c]).std()>0]
betam=[c for c in allm if "aspect_ratio_over" not in c]     # drop derived ratio for the weights
def short_m(c):
    return (c.replace("metrics.","").replace("_over_n_field_periods","/N").replace("rotational_transform","iota")
            .replace("magnetic_","").replace("minimum_normalized_","min_")
            .replace("flux_compression_in_regions_of_bad_curvature","flux_compress"))

# ---------- 1-2 scatter rows ----------
def scatter_row(xcol, scale, xlabel, fname, title):
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,p in zip(axes,list(DIPOLE)):
        s=feas(p); cost=num(s[PROB[p]]).to_numpy(); x=np.abs(num(s[xcol]).to_numpy())*scale
        ok=np.isfinite(x)&np.isfinite(cost)&(cost>0)
        ax.scatter(x[ok],cost[ok],s=22,alpha=0.55,edgecolor="none",color=PCOLOR[p])
        if ok.sum()>2 and (cost[ok]>0).all(): ax.set_yscale("log")
        ax.set_title(f"{SHORT[p]}   rho={spear(x[ok],cost[ok]):+.2f}")
        ax.set_xlabel(xlabel); ax.set_ylabel("achieved coil cost")
    fig.suptitle(title,fontsize=17,y=1.02); fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}",bbox_inches="tight"); plt.close(fig); print("->",fname)

scatter_row(IOTA_E, NFP, "|iota| (edge)", "iota_vs_cost.png",
            "Dipole coil cost vs rotational transform  -  no trend (i=3)")
scatter_row(MIRROR, 1.0, "edge mirror ratio", "mirror_vs_cost.png",
            "Dipole coil cost vs mirror ratio  -  clear driver (i=3)")

# ---------- 3 colormap c ----------
probs=list(PROB)
C=np.array([[spear(num(feas(p)[c]).to_numpy(), num(feas(p)[PROB[p]]).to_numpy()) for c in allm] for p in probs])
L=[short_m(c) for c in allm]
fig,ax=plt.subplots(figsize=(0.95*len(allm)+3,0.62*len(probs)+2.5))
im=ax.imshow(C,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
ax.set_xticks(range(len(L))); ax.set_xticklabels(L,rotation=45,ha="right",fontsize=10)
ax.set_yticks(range(len(probs))); ax.set_yticklabels([SHORT[p] for p in probs])
for i in range(len(probs)):
    for j in range(len(allm)):
        if np.isfinite(C[i,j]): ax.text(j,i,f"{C[i,j]:+.2f}",ha="center",va="center",fontsize=8,
                                        color="white" if abs(C[i,j])>0.5 else "black")
ax.set_title("Spearman(coil cost, equilibrium metric), per problem (i=3)\nred = harder, blue = easier")
fig.colorbar(im,ax=ax,shrink=0.7,label="rho"); fig.tight_layout()
fig.savefig(f"{OUT}/metric_colormap.png",bbox_inches="tight"); plt.close(fig); print("-> metric_colormap.png")

# ---------- 4 multiple-R bar ----------
def multR(cost,Xcols):
    cost=np.asarray(cost,float); cols=[np.asarray(c,float) for c in Xcols]; mask=np.isfinite(cost)
    for c in cols: mask&=np.isfinite(c)
    n=mask.sum(); p=len(cols)
    if n<p+3: return float("nan")
    y=z(rank(cost[mask])); Z=np.column_stack([z(rank(c[mask])) for c in cols])
    beta,*_=np.linalg.lstsq(Z,y,rcond=None); r2=1-((y-Z@beta)**2).sum()/(y**2).sum()
    return float(np.sqrt(max(min(r2,1),0)))
Rs={p:multR(num(feas(p)[PROB[p]]).to_numpy(),[num(feas(p)[c]).to_numpy() for c in allm]) for p in probs}
fig,ax=plt.subplots(figsize=(8.5,5))
ax.bar([SHORT[p] for p in probs],[Rs[p] for p in probs],color=[PCOLOR[p] for p in probs])
for i,p in enumerate(probs):
    if np.isfinite(Rs[p]): ax.text(i,Rs[p]+0.015,f"{Rs[p]:.2f}",ha="center",fontsize=13)
ax.set_ylim(0,1); ax.set_ylabel("R (multiple correlation)")
ax.set_title(f"How well all {len(allm)} equilibrium metrics predict each coil cost (i=3)")
fig.tight_layout(); fig.savefig(f"{OUT}/multiple_R.png"); plt.close(fig); print("-> multiple_R.png")

# ---------- 5 f_phi correlation (raw) ----------
g=feas("G_l2_dipole"); G=pd.DataFrame({"config_id":g["config_id"].values,"fphi":num(g["f_Phi"]).values}) if len(g) else None
if G is not None and len(G):
    fig,axes=plt.subplots(1,2,figsize=(13,5.6))
    for ax,(cp,cobj,tag) in zip(axes,[("C_sparse_dipole","f_l1_Phi","sparse (f_l1_Phi)"),
                                       ("D_thin_dipole","f_max_Phi","thin (f_max_Phi)")]):
        c=feas(cp); Cd=pd.DataFrame({"config_id":c["config_id"].values,"cost":num(c[cobj]).values})
        m=G.merge(Cd,on="config_id").dropna()
        x=m["fphi"].to_numpy(); y=m["cost"].to_numpy(); ok=(x>0)&(y>0)&np.isfinite(x)&np.isfinite(y)
        ax.scatter(x[ok],y[ok],s=26,alpha=0.6,edgecolor="none",color="#6a51a3")
        if ok.sum()>2: ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"f_Phi vs {tag}\nn={ok.sum()}  rho={spear(x[ok],y[ok]):+.2f}")
        ax.set_xlabel("f_Phi achieved"); ax.set_ylabel(f"{tag.split()[0]} achieved")
    fig.suptitle("f_Phi tracks the dipole complexity metrics (i=3)",fontsize=17,y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/fphi_correlation.png",bbox_inches="tight"); plt.close(fig)
    print("-> fphi_correlation.png")

# ---------- 6 plasma-parameter correlation matrix (Rxx) ----------
cfg=df.drop_duplicates("config_id")[allm].apply(num).dropna()
Xr=np.column_stack([rank(cfg[c].to_numpy()) for c in allm]); Rxx=np.corrcoef(Xr.T)
fig,ax=plt.subplots(figsize=(10.5,9)); im=ax.imshow(Rxx,cmap="RdBu_r",vmin=-1,vmax=1)
ax.set_xticks(range(len(L))); ax.set_xticklabels(L,rotation=45,ha="right",fontsize=10)
ax.set_yticks(range(len(L))); ax.set_yticklabels(L,fontsize=10)
for i in range(len(L)):
    for j in range(len(L)): ax.text(j,i,f"{Rxx[i,j]:+.2f}",ha="center",va="center",fontsize=7,
                                     color="white" if abs(Rxx[i,j])>0.55 else "black")
ax.set_title("R: correlation matrix among plasma parameters (rank-based)\nredundant pairs = dark; this is what R$^{-1}$ divides out")
fig.colorbar(im,ax=ax,shrink=0.8,label="rho"); fig.tight_layout()
fig.savefig(f"{OUT}/plasma_corr_matrix.png",bbox_inches="tight"); plt.close(fig); print("-> plasma_corr_matrix.png")
print(f"   cond(Rxx) = {np.linalg.cond(Rxx):.0f}")

# ---------- 7 deconfounded weights (Rxx^-1 c), drop derived ratio ----------
Lb=[short_m(c) for c in betam]; B=np.full((len(probs),len(betam)),np.nan); Rd={}
for i,p in enumerate(probs):
    s=feas(p); cost=num(s[PROB[p]]).to_numpy(); X=np.column_stack([num(s[c]).to_numpy() for c in betam])
    m=np.isfinite(cost)&np.all(np.isfinite(X),axis=1)
    if m.sum()<len(betam)+3: continue
    rc=rank(cost[m]); R=np.column_stack([rank(X[m,j]) for j in range(X.shape[1])])
    rc=(rc-rc.mean())/rc.std(); R=(R-R.mean(0))/R.std(0)
    c=R.T@rc/len(rc); beta=np.linalg.pinv(np.corrcoef(R.T))@c
    B[i]=beta; Rd[p]=float(np.sqrt(max(min(c@beta,1),0)))
vlim=np.nanmax(np.abs(B))
fig,ax=plt.subplots(figsize=(0.95*len(betam)+3,0.62*len(probs)+2.5))
im=ax.imshow(B,cmap="PRGn",vmin=-vlim,vmax=vlim,aspect="auto")
ax.set_xticks(range(len(Lb))); ax.set_xticklabels(Lb,rotation=45,ha="right",fontsize=10)
ax.set_yticks(range(len(probs))); ax.set_yticklabels([SHORT[p] for p in probs])
for i in range(len(probs)):
    for j in range(len(betam)):
        if np.isfinite(B[i,j]): ax.text(j,i,f"{B[i,j]:+.2f}",ha="center",va="center",fontsize=7,
                                        color="white" if abs(B[i,j])>0.6*vlim else "black")
ax.set_title("R$^{-1}$c: deconfounded weights per problem (standardized regression)\n"
             "the colormap with plasma-parameter collinearity divided out, iota goes to ~0")
fig.colorbar(im,ax=ax,shrink=0.7,label="weight"); fig.tight_layout()
fig.savefig(f"{OUT}/deconfounded_weights.png",bbox_inches="tight"); plt.close(fig); print("-> deconfounded_weights.png")

# ---------- printed summary ----------
print("\nmultiple-R (cost ~ all metrics):", {SHORT[p]:round(Rs[p],2) for p in probs if np.isfinite(Rs[p])})
print("R from c^T Rxx^-1 c            :", {SHORT[p]:round(Rd[p],2) for p in probs if p in Rd})
print(f"\ndone -> {OUT}/  (7 figures)")