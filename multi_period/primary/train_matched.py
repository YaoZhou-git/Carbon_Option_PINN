"""Two bounded warm starts; reuse the existing architecture, sampling and PDE loss."""
from pathlib import Path
import sys,json,time,csv,hashlib
from dataclasses import replace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/matched_training'
CODE=ROOT/'core'
MODELS=ROOT/'models/matched'
sys.path.insert(0,str(CODE))
from factor_model import Model,factors
from factor_pinn import FactorPINN,sample,residual

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    MODELS.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('Matched training requires the existing CUDA resource; retain Sobol if unavailable.')
    source=ROOT/'models/baseline/N4_factor_pinn.pt'
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    device=torch.device('cuda')
    scenarios=[('flat',1.9774154606995185,[60,60,60,60],14.571845890486031),
               ('upward',3.458803407034734,[60,65,70,75],13.340157346614122)]
    results=[]
    for index,(name,mult,quotes,reference) in enumerate(scenarios):
        dest=MODELS/f'N4_matched_{name}.pt'
        cfg=replace(Model(),beta=tuple(.8*mult for _ in range(4)))
        torch.manual_seed(20261111+index);np.random.seed(20261111+index)
        net=FactorPINN(cfg,4).to(device)
        net.load_state_dict(checkpoint['state_dict'])
        # load_state_dict also restores buffers: replace the old beta explicitly.
        with torch.no_grad():net.beta.copy_(torch.tensor(cfg.beta,device=device))
        assert np.allclose(net.beta.detach().cpu().numpy(),cfg.beta)
        epochs=2500;batch=2048
        opt=torch.optim.Adam(net.parameters(),lr=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=1e-5)
        history=[];torch.cuda.synchronize();start=time.perf_counter()
        for epoch in range(1,epochs+1):
            opt.zero_grad(set_to_none=True)
            state=sample(net,batch,device)
            loss=residual(net,state).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite PDE residual')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step();scheduler.step()
            if epoch==1 or epoch%500==0:
                rec={'scenario':name,'epoch':epoch,'PDE_loss':loss.item()}
                history.append(rec);print(json.dumps(rec),flush=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        state=torch.tensor(np.r_[factors(quotes,cfg)/cfg.pi,0.][None],device=device,dtype=torch.float32)
        with torch.no_grad():price=net(state).item()*cfg.pi
        torch.save({'state_dict':net.state_dict(),'N':4,'parameters':cfg.__dict__,
                    'warm_start':str(source),'warm_start_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                    'epochs':epochs,'batch':batch,'seed':20261111+index,
                    'learning_rate':[1e-4,1e-5],'seconds':seconds,
                    'device':torch.cuda.get_device_name(0),'data_labels_used':False},dest)
        (OUT/f'matched_{name}_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        rec={'curve':name,'beta_multiplier':mult,'beta':cfg.beta[0],'PINN_price':price,
             'Sobol_reference':reference,'difference':price-reference,
             'relative_error_percent':100*abs(price-reference)/reference,'training_seconds':seconds,
             'additional_epochs':epochs,'checkpoint':str(dest.relative_to(ROOT)),
             'checkpoint_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
        results.append(rec)
        (OUT/'matched_training_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(json.dumps(rec),flush=True)
    with (OUT/'matched_pinn_prices.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)

if __name__=='__main__':run()
