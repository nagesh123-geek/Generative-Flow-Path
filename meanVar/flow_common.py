"""Shared architectures and flow paths for TRP-Cage flow matching."""
import torch
import torch.nn as nn

DATA_DIM = 38
HIDDEN_DIM = 256
DROPOUT = 0.2

class TimeEmbedding(nn.Module):
    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
    def forward(self, t):
        return self.net(t.reshape(-1, 1).float())

class MLP(nn.Module):
    """Plain MLP, without residual connections."""
    def __init__(self, data_dim=DATA_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.time_embed = TimeEmbedding(hidden_dim)
        self.block1 = nn.Sequential(nn.Linear(data_dim+hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.block2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.block3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.block4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden_dim, data_dim)
    def forward(self, x, t):
        h = torch.cat([x, self.time_embed(t)], dim=-1)
        return self.out(self.block4(self.block3(self.block2(self.block1(h)))))

class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim,dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim,dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout))
    def forward(self, x):
        return x + self.block(x)

class ResidualMLP(nn.Module):
    """MLP with residual blocks."""
    def __init__(self, data_dim=DATA_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.time_embed = TimeEmbedding(hidden_dim)
        self.input_layer = nn.Sequential(nn.Linear(data_dim+hidden_dim,hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.res1, self.res2, self.res3, self.res4 = (ResBlock(hidden_dim,dropout) for _ in range(4))
        self.out = nn.Linear(hidden_dim,data_dim)
    def forward(self,x,t):
        h=self.input_layer(torch.cat([x,self.time_embed(t)],dim=-1))
        return self.out(self.res4(self.res3(self.res2(self.res1(h)))))

class TransformerFM(nn.Module):
    """Transformer architecture matching the supplied code's global-vector token design."""
    def __init__(self, data_dim=DATA_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, num_layers=6, num_heads=8):
        super().__init__()
        self.time_embed=TimeEmbedding(hidden_dim)
        self.input_proj=nn.Sequential(nn.Linear(data_dim,hidden_dim),nn.LayerNorm(hidden_dim))
        layer=nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=num_heads,dim_feedforward=4*hidden_dim,dropout=dropout,activation='relu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=num_layers)
        self.output_proj=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,data_dim))
    def forward(self,x,t):
        h=self.input_proj(x)+self.time_embed(t)
        h=self.encoder(h.unsqueeze(1)).squeeze(1)
        return self.output_proj(h)

def build_model(architecture, data_dim=DATA_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
    key=architecture.lower().replace('-','_')
    if key in ('mlp','plain','no_residual'):
        return MLP(data_dim,hidden_dim,dropout)
    if key in ('residual','res_mlp','mlp_residual','mlp_rs'):
        return ResidualMLP(data_dim,hidden_dim,dropout)
    if key in ('transformer','trans'):
        return TransformerFM(data_dim,hidden_dim,dropout)
    raise ValueError(f'Unknown architecture: {architecture}. Choose mlp, residual, transformer.')

# Every path uses t=0 as noise and t=1 as data.
def make_path(flow_name, x, z, t):
    name=flow_name.lower()
    t=t.reshape(-1,1)
    if name=='linear':
        xt=t*x+(1-t)*z
        ut=x-z
    elif name=='trigonometric':
        a=(torch.pi/2)*t
        xt=torch.sin(a)*x+torch.cos(a)*z
        ut=(torch.pi/2)*(torch.cos(a)*x-torch.sin(a)*z)
    elif name=='operator':
        # Fixed, feature-dependent diagonal operators; endpoint-preserving schedules.
        p=torch.linspace(1.0,2.0,x.shape[1],device=x.device,dtype=x.dtype).view(1,-1)
        q=torch.linspace(2.0,1.0,x.shape[1],device=x.device,dtype=x.dtype).view(1,-1)
        tc=t.clamp(0.0,1.0)
        a=tc.pow(p); b=(1-tc).pow(q)
        da=p*tc.pow(p-1.0); db=-q*(1-tc).pow(q-1.0)
        xt=a*x+b*z
        ut=da*x+db*z
    elif name=='spherical':
        # Slerp directions with endpoint magnitudes retained. Near parallel vectors use linear limit.
        eps=1e-8
        xn=x/torch.linalg.norm(x,dim=1,keepdim=True).clamp_min(eps)
        zn=z/torch.linalg.norm(z,dim=1,keepdim=True).clamp_min(eps)
        theta=torch.acos((xn*zn).sum(1,keepdim=True).clamp(-1+1e-7,1-1e-7))
        st=torch.sin(theta)
        small=st.abs()<1e-4
        den=torch.where(small,torch.ones_like(st),st)
        a=torch.sin((1-t)*theta)/den
        b=torch.sin(t*theta)/den
        da=-theta*torch.cos((1-t)*theta)/den
        db=theta*torch.cos(t*theta)/den
        xt=torch.where(small,(1-t)*z+t*x,a*z+b*x)
        ut=torch.where(small,x-z,da*z+db*x)
    else:
        raise ValueError(f'Unknown flow: {flow_name}')
    return xt,ut

def velocity(model, x, t):
    return model(x, t)
