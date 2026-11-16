"""Generate samples using a trained operator flow checkpoint."""
import argparse, os
import numpy as np
import torch
from flow_common import build_model, DATA_DIM

@torch.no_grad()
def sample(model,n_samples,steps,device):
    model.eval(); x=torch.randn(n_samples,DATA_DIM,device=device); dt=1.0/steps
    # Euler ODE integration from t=0 (noise) to t=1 (data).
    for i in range(steps):
        t=torch.full((n_samples,),i/steps,device=device)
        x=x+dt*model(x,t)
    return x.cpu().numpy()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='runs/operator/model_latest.pt')
    p.add_argument('--n-samples',type=int,default=500000)
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=2048)
    p.add_argument('--device',default='cuda:1' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',default='runs/operator/generated_samples_trans.npy')
    args=p.parse_args(); device=torch.device(args.device)
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    if ckpt.get('flow')!='operator': raise ValueError(f"Checkpoint flow is {ckpt.get('flow')}, expected operator")
    model=build_model(ckpt['architecture'],ckpt.get('data_dim',DATA_DIM),ckpt.get('hidden_dim',256),ckpt.get('dropout',0.2)).to(device)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    pieces=[]
    for n in range(0,args.n_samples,args.batch_size):
        count=min(args.batch_size,args.n_samples-n)
        pieces.append(sample(model,count,args.steps,device))
        print(f'Generated {min(n+count,args.n_samples)}/{args.n_samples}',flush=True)
    generated=np.concatenate(pieces,axis=0)
    os.makedirs(os.path.dirname(args.output) or '.',exist_ok=True)
    np.save(args.output,generated)
    print('Saved',args.output,'shape=',generated.shape)
if __name__=='__main__': main()
