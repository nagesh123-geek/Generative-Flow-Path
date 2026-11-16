"""Train the linear flow with MLP, residual MLP, or Transformer architecture."""
import argparse, os, time
import numpy as np
import torch
import torch.nn.functional as F
from flow_common import build_model, make_path, DATA_DIM

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='trpcage.npy',help='NumPy array with shape (N,38)')
    p.add_argument('--architecture',choices=['mlp','residual','transformer'],default='mlp')
    p.add_argument('--epochs',type=int,default=50000)
    p.add_argument('--batch-size',type=int,default=4096)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--hidden-dim',type=int,default=256)
    p.add_argument('--dropout',type=float,default=0.2)
    p.add_argument('--save-every',type=int,default=10000)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--outdir',default='runs/linear')
    args=p.parse_args()
    device=torch.device(args.device)
    arr=np.load(args.data).astype(np.float32)
    if arr.ndim!=2 or arr.shape[1]!=DATA_DIM: raise ValueError(f'Expected (N,38), got {arr.shape}')
    if not np.isfinite(arr).all(): raise ValueError('Data contains NaN or infinity')
    data=torch.from_numpy(arr)
    os.makedirs(args.outdir,exist_ok=True)
    model=build_model(args.architecture,DATA_DIM,args.hidden_dim,args.dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    loss_log=[]; start=time.time()
    print(f'Flow=linear architecture={args.architecture} data={arr.shape} device={device}',flush=True)
    for epoch in range(1,args.epochs+1):
        model.train(); perm=torch.randperm(len(data)); total=0.0; seen=0
        for idx in perm.split(args.batch_size):
            x=data[idx].to(device,non_blocking=True); z=torch.randn_like(x)
            t=torch.rand(x.shape[0],device=device)
            xt,target=make_path('linear',x,z,t)
            pred=model(xt,t); loss=F.mse_loss(pred,target)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total += loss.item()*len(x); seen += len(x)
        avg=total/seen; loss_log.append(avg)
        if epoch==1 or epoch%10==0: print(f'Epoch {epoch}/{args.epochs} loss={avg:.7f}',flush=True)
        np.savetxt(os.path.join(args.outdir,'loss.txt'),np.asarray(loss_log))
        if epoch%args.save_every==0 or epoch==args.epochs:
            ckpt={'flow':'linear','architecture':args.architecture,'model_state_dict':model.state_dict(),'data_dim':DATA_DIM,'hidden_dim':args.hidden_dim,'dropout':args.dropout,'epoch':epoch,'loss':avg}
            path=os.path.join(args.outdir,f'model_epoch_{epoch:06d}.pt'); torch.save(ckpt,path)
            torch.save(ckpt,os.path.join(args.outdir,'model_latest.pt'))
            print('Saved',path,flush=True)
    print(f'Training finished in {(time.time()-start)/60:.1f} min',flush=True)
if __name__=='__main__': main()
