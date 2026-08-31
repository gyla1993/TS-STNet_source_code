import torch
import torch.nn as nn
import torch.nn.functional as F

class Improved_Complex_Attention(nn.Module):
    def __init__(self, args, K):
        super(Improved_Complex_Attention, self).__init__()
        self.d_model = args.d_model
        self.K=K
        self.scale = self.d_model ** -0.5

        # 多头注意力
        self.num_heads = 8
        self.head_dim = self.d_model // self.num_heads

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key_value):
        # query: B,N,k,d_model

        B, N,k, d_model = query.shape

        # 重塑为多头
        Q = self.W_q(query).reshape(B, N*k, self.num_heads, self.head_dim)
        K = self.W_k(key_value).reshape(B, N*k, self.num_heads, self.head_dim)
        V = self.W_v(key_value).reshape(B, N*k, self.num_heads, self.head_dim)

        # 转置以便矩阵乘法
        Q = Q.permute(0,2,1,3)  # B,num_heads,N*k,head_dim
        K = K.permute(0,2,1,3)  # B,num_heads,N*k,head_dim
        V = V.permute(0,2,1,3)  # B,num_heads,N*k,head_dim

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))#B,num_heads,N*k,N*k
        attn_scores = attn_scores * self.scale

        # 注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        output = torch.matmul(attn_weights, V)
        output = output.permute(0,2,1,3).reshape(B, N,k, d_model)

        output = self.W_o(output)

        return output, attn_weights.mean(dim=2)  # 返回平均注意力权重

class ComplexReLU(nn.Module):
    """复数ReLU激活函数"""

    def __init__(self):
        super(ComplexReLU, self).__init__()

    def forward(self, x):
        # 对实部和虚部分别应用ReLU
        real = F.relu(x.real)
        imag = F.relu(x.imag)
        return torch.complex(real, imag)


class ComplexLayerNorm(nn.Module):
    """复数层归一化"""

    def __init__(self, normalized_shape, eps=1e-5):
        super(ComplexLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight_real = nn.Parameter(torch.ones(normalized_shape))
        self.weight_imag = nn.Parameter(torch.ones(normalized_shape))
        self.bias_real = nn.Parameter(torch.zeros(normalized_shape))
        self.bias_imag = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # 对实部和虚部分别进行层归一化
        real = F.layer_norm(x.real, self.normalized_shape, eps=self.eps)
        imag = F.layer_norm(x.imag, self.normalized_shape, eps=self.eps)

        real = real * self.weight_real + self.bias_real
        imag = imag * self.weight_imag + self.bias_imag

        return torch.complex(real, imag)


class ComplexSigmoid(nn.Module):
    """复数Sigmoid激活函数"""

    def __init__(self):
        super(ComplexSigmoid, self).__init__()

    def forward(self, x):
        # 对实部和虚部分别应用Sigmoid
        real = torch.sigmoid(x.real)
        imag = torch.sigmoid(x.imag)
        return torch.complex(real, imag)

class ComplexDropout(nn.Module):
    """复数Dropout"""

    def __init__(self, p=0.1):
        super(ComplexDropout, self).__init__()
        self.p = p
        self.dropout_real = nn.Dropout(p)
        self.dropout_imag = nn.Dropout(p)

    def forward(self, x):
        real = self.dropout_real(x.real)
        imag = self.dropout_imag(x.imag)
        return torch.complex(real, imag)

class Frequency_Aware_Attn(nn.Module):
    def __init__(self, args, K):
        super(Frequency_Aware_Attn, self).__init__()
        self.seq_len = args.seq_len//2+ 1  # rfft后长度
        self.device = args.device
        self.d_model = args.d_model
        self.K=K
        self.pred_len = args.pre_len
        # 复数操作
        self.relu=ComplexReLU()
        self.complex_dropout = ComplexDropout(0.1)
        self.layernorm=ComplexLayerNorm(self.d_model)
        self.sigmoid=ComplexSigmoid()

        # 2. 添加层归一化
        self.layernorm1 = nn.LayerNorm(self.d_model)
        self.layernorm2 = nn.LayerNorm(self.d_model)

        # 3. 改进的投影层，添加残差连接
        self.fc1 = nn.Linear(self.seq_len, self.d_model).to(torch.cfloat)
        self.fc2 = nn.Linear(self.d_model, self.d_model).to(torch.cfloat)
        self.fc3 = nn.Linear(self.d_model, self.seq_len).to(torch.cfloat)
        # 5. 改进的注意力机制
        self.real_attn = Improved_Complex_Attention(args, self.K)
        self.imag_attn = Improved_Complex_Attention(args, self.K)

        # 6. 输出层改进
        self.output_proj = nn.Sequential(
            nn.Linear(self.K * self.pred_len, 4 * self.pred_len),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(4* self.pred_len, self.pred_len)
        )

        # 7. 频域权重学习
        self.freq_weights = nn.Parameter(torch.ones(self.seq_len))
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化"""
        # Xavier初始化用于线性层
        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # 频域权重初始化
        nn.init.constant_(self.freq_weights, 1.0)

    def forward(self, x):
        B, N,k,L = x.shape
        xp_freq = torch.fft.rfft(x, dim=-1)  # B,N,k,L+1
        freq_weights = torch.sigmoid(self.freq_weights)
        xp_freq = xp_freq * freq_weights.view(1, 1, 1, -1)#B,N,k,L+1
        # 编码频域特征
        xp_flat = xp_freq.reshape(-1, self.seq_len)
        xp_emb = self.fc1(xp_flat)  # (B*N*k), d_model
        xp_emb = self.relu(xp_emb)
        xp_emb = self.fc2(xp_emb)  # (B*N*k), d_model
        xp_emb = xp_emb.reshape(B,N,k, -1)#B,N,k,D
        #实数域和复数域attn
        xp_emb_real, xp_emb_imag = xp_emb.real, xp_emb.imag

        output_real, A_real = self.real_attn(xp_emb_real, xp_emb_real)
        output_imag, A_imag = self.imag_attn(xp_emb_imag, xp_emb_imag)

        # 残差连接
        output_real = output_real + xp_emb_real
        output_imag = output_imag + xp_emb_imag

        output = torch.complex(output_real, output_imag)

        #  输出投影
        output = self.fc3(output)  # B,N,k,seq_len
        output = output.reshape(B,N,k, self.seq_len)

        # 逆变换
        output_time = torch.fft.irfft(output, n=L, dim=-1)#B,N,k,L

        # 输出层
        output_flat = output_time.reshape(B,N,-1)
        output_final = self.output_proj(output_flat)
        output_final = output_final.reshape(B, N,1, self.pred_len)

        return output_final