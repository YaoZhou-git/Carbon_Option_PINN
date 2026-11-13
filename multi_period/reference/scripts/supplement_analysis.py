"""Predefined dependence diagnostics and precision checks for small effects."""
from run_analysis import *

def matrix_estimate(a,c,R,power=16,reps=8):
    M,V=trans(replace(c,rho=.3),4)
    V=V.copy()
    for i in range(4):
        for j in range(i+1,4):V[i,j]=V[j,i]=V[i,j]/.3*R[i,j]
    y=M*ndtri(xfactor(a,c)/c.pi)+normals(power,reps,4)@np.linalg.cholesky(V).T
    s=c.pi*ndtr(y)@weights(c,4)
    ps=np.exp(-c.r*c.tau)*np.maximum(s-c.K,0).mean(axis=1)
    return float(ps.mean()),float(ps.std(ddof=1)/np.sqrt(reps))

rows=[]
for i in range(4):
    for j in range(i+1,4):
        for rho in [-.3,0.,.3,.6,.9]:
            R=np.full((4,4),.3);np.fill_diagonal(R,1);R[i,j]=R[j,i]=rho
            eig=float(np.linalg.eigvalsh(R).min())
            if eig<=0:
                rows.append({'pair':f'{i+1}-{j+1}','correlation':rho,'min_eigenvalue':eig,'C':'','se':'','valid':False})
            else:
                c,se=matrix_estimate(A,BASE,R)
                rows.append({'pair':f'{i+1}-{j+1}','correlation':rho,'min_eigenvalue':eig,'C':c,'se':se,'valid':True})
write('pairwise_correlation',rows)
rows=[]
for label,rho,K,j in [('下倾',-.25,50.,3),('平坦',-.25,70.,3),('上倾',-.25,70.,3),('下倾',.7,70.,2)]:
    c=replace(BASE,rho=rho,K=K)
    for h in [.1,.05,.025]:
        d,se=delta(CURVES[label],c,j,h=h,power=18)
        rows.append({'curve':label,'rho':rho,'K':K,'period':j+1,'h':h,'delta':d,'se':se,'halfwidth95':2.364624*se})
write('counterexample_refinement',rows)
# Validate history's fixed-maturity PINN outputs against NEW independent integration.
import pandas as pd
saved=pd.read_csv(DATA/'saved_PINN_input.csv')
rows=[]
for row in saved.itertuples():
    a=[float(row.A1),45.,30.,20.][:int(row.N)]
    p=estimate(a,power=17)
    rows.append({'N':int(row.N),'A1':row.A1,'saved_PINN':row.PINN_C,'new_QMC':p['C'],
        'QMC_halfwidth95':p['halfwidth95'],'PINN_minus_new_QMC':row.PINN_C-p['C'],
        'relative_absolute_difference':abs(row.PINN_C-p['C'])/max(p['C'],1e-10),
        'discounted_lower_bound':np.exp(-.15)*max(row.A1-50,0),'upper_bound':np.exp(-.15)*row.A1})
write('pinn_independent_recheck',rows)
print('Supplement completed',flush=True)
