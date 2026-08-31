import torch.nn as nn
import torch
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class gcn(nn.Module):
    """Graph convolution network."""

    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a.to(x.device))
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a.to(x.device))
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class HierarchicalGCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len, d_model,pred_len,order=2):
        super(HierarchicalGCN, self).__init__()
        self.gcn=gcn(c_in, c_out, dropout, support_len, order=2)
        self.fc = linear(c_out,1)
        self.fc_1=nn.Linear(d_model,pred_len)

    def forward(self,x,new_adp):
        #x:B,D,N,1
        x=x.permute(0,3,2,1)#x:B,1,N,D
        spatio_emb=self.gcn(x,new_adp)
        out=self.fc(spatio_emb)#B,1,N,D
        out=self.fc_1(out)
        return out