"""TF-STNet architecture used for the reported wind-speed experiment."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch.nn as nn
from .layers.tcn import GateTCN
import numpy as np
from .layers.frequency import Frequency_Aware_Attn
from .layers.gcn import HierarchicalGCN
import torch

class find_k_nearest_neighbors(nn.Module):
    def __init__(self, k,device):
        super(find_k_nearest_neighbors, self).__init__()
        self.k=k
        self.device=device

    def forward(self,csta,cnwp,nwp_his,nwp_fut):
        #csta:N,2
        #cnwp:lat,lon,2
        #nwp_his:B,lat,lon,L
        #nwp_fut:B,lat,lon,L
        from scipy.spatial.distance import cdist
        B,lat,lon,L=nwp_his.shape
        grid_coords = cnwp.reshape(-1, 2)  # 变形为(lat*lon, 2)，每个网格的坐标
        # 计算站点csta与网格grid_coords之间的距离
        distances = cdist(csta, grid_coords, metric='euclidean')  # 计算站点与网格点的欧氏距离
        nearest_indices = np.argsort(distances, axis=1)[:, :self.k]  # 选择k个最近的网格点
        cnwp_k=grid_coords[nearest_indices, :]#N,K,2
        nwp_his_k=nwp_his.reshape(B,-1,L)[:,nearest_indices,:]#B,N,K,L
        nwp_fut_k=nwp_fut.reshape(B,-1,L)[:,nearest_indices,:]
        return cnwp_k,nwp_his_k,nwp_fut_k


class Model(nn.Module):
    def __init__(self, args, predefined_A):
        super(Model, self).__init__()
        self.in_dim=args.in_dim
        self.out_dim=args.d_model
        self.pred_len=args.pre_len
        self.num_nodes=args.num_nodes
        self.d_model=args.d_model
        self.k=args.k
        self.device=args.device
        self.obs_timeTCN=GateTCN(in_dim=self.in_dim,out_dim=self.out_dim,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=8,layers=2)
        self.nwp_timeTCN=GateTCN(in_dim=self.in_dim,out_dim=self.out_dim,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=8,layers=2)
        self.spationGCN=HierarchicalGCN( c_in=self.in_dim, c_out=self.d_model, dropout=0.1, support_len=2, d_model=self.d_model,pred_len=self.pred_len,order=2)
        self.nwp_fft_spa=Frequency_Aware_Attn(args, self.k)
        self.find_k_nearest_neighbors = find_k_nearest_neighbors(self.k, self.device)
        self.k_fusion = nn.Sequential(
            nn.Linear((self.k+1)*self.num_nodes, 2*self.num_nodes),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(2*self.num_nodes,self.num_nodes)
        )
        self.adj=predefined_A
        self.predict_layer=nn.Conv2d(2,1,(1,1))

    def forward(self,obs_his,nwp_his,nwp_fut,his_mark,fut_mark,csta,cnwp):
        '''
        Input:obs_his:(B,N,L)
              era_his:(B,lat,lon,L)
              pan_fut:(B,lat,lon,L)
              cobs:(24,2)
              cera:(lat,lon,2):25,37,2
              cpan:(lat,lon,2)25,36,2
        '''

        cnwp_k,nwp_his_k,nwp_fut_k=self.find_k_nearest_neighbors(csta,cnwp,nwp_his,nwp_fut)#cnwp_k:N,k,2,B,N,k,L
        obs_his=obs_his.unsqueeze(1)#B,1,N,L
        B,_,N,L=obs_his.shape
        nwp_fut=nwp_fut_k.reshape(B,N*self.k,L).unsqueeze(1)#B,1,N*k,L
        obs_time_emb=self.obs_timeTCN(obs_his)#B,D,N,1
        nwp_time_emb=self.nwp_timeTCN(nwp_fut)#B,D,N*k,1
        time_emb=torch.cat((obs_time_emb,nwp_time_emb),dim=-2).squeeze(-1)#B,D,N(k+1)
        tilde_time_emb=self.k_fusion(time_emb).unsqueeze(-1)#B,D,N,1
        obs_emb=self.spationGCN(tilde_time_emb,self.adj)#B,1,N,L
        nwp_fft_emb=self.nwp_fft_spa(nwp_fut_k).permute(0,2,1,3)#B,1,N,L
        emb=torch.cat([obs_emb,nwp_fft_emb],dim=1)
        output=self.predict_layer(emb)#B,1,N,L

        return output.squeeze(1)


@dataclass(frozen=True)
class TFSTNetConfig:
    """Architecture constants matching the released checkpoint."""

    num_nodes: int = 60
    history_len: int = 24
    prediction_len: int = 24
    d_model: int = 64
    k: int = 2


class TFSTNet(Model):
    """Named public interface for the released TF-STNet model."""

    def __init__(self, supports: tuple[torch.Tensor, torch.Tensor], device: torch.device):
        config = TFSTNetConfig()
        args = SimpleNamespace(
            in_dim=1,
            d_model=config.d_model,
            pre_len=config.prediction_len,
            seq_len=config.history_len,
            num_nodes=config.num_nodes,
            k=config.k,
            device=device,
        )
        super().__init__(args, list(supports))
