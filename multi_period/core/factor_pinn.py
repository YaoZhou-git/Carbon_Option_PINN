"""Factor-coordinate PINN; hard terminal payoff, soft PDE constraint.

The smooth Bachelier term is an analytic trial function, not a price label.
Gaussian quadrature is used only for independent post-training diagnostics.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from factor_model import Model,matrix,factors,value

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(4)


def phi(x):return torch.exp(-x.square()/2)/np.sqrt(2*np.pi)
def Phi(x):return .5*(1+torch.erf(x/np.sqrt(2)))


class FactorPINN(nn.Module):
    def __init__(self,cfg,n):
        super().__init__();self.cfg=cfg;self.n=n
        self.register_buffer('weights',torch.tensor(matrix(cfg,n)[0],dtype=torch.float32))
        R=np.full((n,n),cfg.rho);np.fill_diagonal(R,1)
        self.register_buffer('R',torch.tensor(R,dtype=torch.float32))
        self.register_buffer('T',torch.tensor(cfg.T[:n],dtype=torch.float32))
        self.register_buffer('alpha',torch.tensor(cfg.alpha[:n],dtype=torch.float32))
        self.register_buffer('beta',torch.tensor(cfg.beta[:n],dtype=torch.float32))
        layers=[nn.Linear(n+1,64),nn.Tanh()]
        for _ in range(2):layers.extend([nn.Linear(64,64),nn.Tanh()])
        layers.append(nn.Linear(64,1));self.net=nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def diffusion(self,state):
        y=state[:,:self.n];t=state[:,self.n:]
        xi=np.sqrt(2)*torch.erfinv(2*y.clamp(1e-6,1-1e-6)-1)
        return phi(xi)*torch.sqrt(self.beta*(self.T-t)**(-self.alpha))

    def forward(self,state):
        y=state[:,:self.n];t=state[:,self.n:];h=self.cfg.tau-t
        d=(y@self.weights)[:,None]-self.cfg.K/self.cfg.pi
        sig=self.diffusion(state)*self.weights
        v=(sig*(sig@self.R)).sum(1,keepdim=True)*h
        sd=torch.sqrt(v.clamp_min(0)+1e-10)
        z=d/sd
        baseline=d*Phi(z)+sd*phi(z)
        inp=torch.cat([2*y-1,2*t/self.cfg.tau-1],1)
        trial=torch.exp(-self.cfg.r*h)*(baseline+h*.1*self.net(inp))
        # Remove the small interior smoothing regularizer at the endpoint;
        # it otherwise gives a nonzero payoff exactly at the strike.
        return torch.where(h==0,d.relu(),trial)


def residual(net,state):
    state=state.detach().requires_grad_(True)
    c=net(state)
    grad=torch.autograd.grad(c.sum(),state,create_graph=True)[0]
    sig=net.diffusion(state)
    out=grad[:,net.n:]-net.cfg.r*c
    for i in range(net.n):
        second=torch.autograd.grad(grad[:,i].sum(),state,create_graph=True)[0]
        for j in range(net.n):
            out=out+.5*(sig[:,i]*sig[:,j]*net.R[i,j]*second[:,j])[:,None]
    return out


def sample(net,count,device):
    y=torch.rand(count,net.n,device=device)*.996+.002
    # Half of collocation points resolve the economically relevant payoff kink.
    if net.n>1:
        m=count//2
        y[:m,1:]*=.45
        target=.5+torch.randn(m,device=device)*.16
        y[:m,0]=(target-y[:m,1:]@net.weights[1:]).clamp(.003,.997)
    # Degenerate factor faces are included; no artificial A=infinity boundary.
    for j in range(net.n):
        s=count//(20*net.n)
        y[j*s:(j+1)*s,j]=1e-4
        y[(net.n+j)*s:(net.n+j+1)*s,j]=1-1e-4
    t=torch.rand(count,1,device=device)*net.cfg.tau*.999
    return torch.cat([y,t],1)


def evaluate(net,cfg,n,device):
    rows=[]
    for a1 in (40.,50.,60.,70.,80.,90.):
        a=[a1,45.,30.,20.][:n]
        x=factors(a,cfg)/cfg.pi
        state=torch.tensor(np.array([np.r_[x,0]]),dtype=torch.float32,device=device)
        with torch.no_grad():c=net(state).item()*cfg.pi
        exact=value(a,cfg,order=40,inner=160)
        lower=np.exp(-cfg.r*cfg.tau)*max(a1-cfg.K,0)
        rows.append({'N':n,'A1':a1,'PINN_C':c,'Gaussian_C':exact,'difference':c-exact,
                     'lower_bound':lower,'upper_bound':np.exp(-cfg.r*cfg.tau)*a1,
                     'lower_bound_shortfall':max(lower-c,0)})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=10000)
    p.add_argument('--batch',type=int,default=2048);p.add_argument('--dimensions',type=int,nargs='+',default=[1,2,3,4])
    args=p.parse_args();cfg=Model();out=ROOT.parent/'models'/'baseline';out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result=[];history=[];timing={}
    for n in args.dimensions:
        torch.manual_seed(20260917+n);np.random.seed(20260917+n)
        net=FactorPINN(cfg,n).to(device)
        opt=torch.optim.Adam(net.parameters(),lr=1e-3)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=5e-5)
        start=time.time()
        for epoch in range(1,args.epochs+1):
            opt.zero_grad(set_to_none=True)
            state=sample(net,args.batch,device)
            rr=residual(net,state)
            loss=rr.square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite PINN residual')
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step();scheduler.step()
            if epoch%500==0 or epoch==1:
                rows=evaluate(net,cfg,n,device)
                diag={'N':n,'epoch':epoch,'PDE_loss':loss.item(),
                      'slice_max_abs_price_difference':max(abs(x['difference']) for x in rows),
                      'reference_C':next(x['PINN_C'] for x in rows if x['A1']==60)}
                history.append(diag);print(json.dumps(diag),flush=True)
        timing[n]=time.time()-start
        result.extend(evaluate(net,cfg,n,device))
        torch.save({'state_dict':net.state_dict(),'N':n,'parameters':cfg.__dict__,
                    'epochs':args.epochs,'batch':args.batch,'device':str(device)},out/f'N{n}_factor_pinn.pt')
        for name,rows in [('price_slice',result),('training_history',history)]:
            with (out/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        (out/'training_metadata.json').write_text(json.dumps({'seconds_by_N':timing,'device':str(device),
           'epochs':args.epochs,'batch':args.batch,'data_labels_used':False,
           'terminal_condition':'hard enforced','input':'bounded recursive factors X/pi and valuation time'},indent=2),encoding='utf-8')
    print('Factor PINN training completed',flush=True)


if __name__=='__main__':main()
