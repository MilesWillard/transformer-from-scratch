import math
import torch
from torch import nn

TOKENS = 2_000
DIM = 128
LAYERS = 4
HEADS = 4
DK = 32
DV = 32
HIDDEN = [512]

class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, eps=1e-6):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        ctx.save_for_backward(x, gamma, rms)
        return x / rms * gamma

    @staticmethod
    def backward(ctx, g):
        x, gamma, rms = ctx.saved_tensors
        d = x.size(-1)
        gg = g * gamma
        grad_x = gg / rms - x * (gg * x).sum(-1, keepdim=True) / (d * rms**3)
        grad_gamma = (g * x / rms).reshape(-1, d).sum(0).reshape(gamma.shape)
        return grad_x, grad_gamma, None

class SmoothReLU(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.clamp_min(x,0)+torch.log1p(torch.exp(-torch.abs(x)))
    
    @staticmethod
    def setup_context(ctx,x,y):
        ins, = x
        ctx.save_for_backward(ins)
        
    @staticmethod
    def backward(ctx,grad_out):
        x, = ctx.saved_tensors
        return grad_out * 1/(1+torch.exp(-x))
    
class SoftMax(torch.autograd.Function):
    @staticmethod
    def forward(x,dim=-1):
        shifted = x - torch.max(x,dim=dim,keepdim=True).values
        exp_x = torch.exp(shifted)
        return exp_x/torch.sum(exp_x,dim=dim,keepdim=True)
    
    @staticmethod
    def setup_context(ctx,inputs,outputs):
        dim = inputs[1]
        ctx.save_for_backward(outputs)
        ctx.dim = dim
        
    @staticmethod
    def backward(ctx,grad_out):
        output, = ctx.saved_tensors
        dim = ctx.dim
        grad_input = output * (grad_out - (grad_out * output).sum(dim=dim, keepdim=True))
        return grad_input, None
    
class PositionalEncoding(nn.Module):
    def forward(self, x):
        _, context, dim = x.shape
        positions = torch.arange(context, device=x.device, dtype=torch.float32)[:, None]
        dimensions = torch.arange(dim, device=x.device, dtype=torch.int64)[None, :]
        
        angles = positions / (10000.0 ** (2 * (dimensions // 2) / dim))
        pe = torch.where(dimensions % 2 == 0, torch.sin(angles), torch.cos(angles))
        
        return (x * math.sqrt(dim)) + pe.to(dtype=x.dtype)
    
class QKVProjection(torch.autograd.Function):
    @staticmethod
    def forward(x,w_q,w_k,w_v):
        y = x[:,None,:,:]
        return y @ w_q, y @ w_k, y @ w_v
    
    @staticmethod
    def setup_context(ctx,inputs,outputs):
        ctx.save_for_backward(*inputs)
        
    @staticmethod
    def backward(ctx, *grad_out):
        x, w_q, w_k, w_v = ctx.saved_tensors
        y = x[:,None,:,:]
        grad_x = torch.zeros_like(x)
        grad_weights = []
        for i, w in enumerate((w_q,w_k,w_v)):
            grads = grad_out[i]
            grad_x += (grads @ torch.transpose(w, -2, -1)).sum(dim=1)
            grad_weights.append(torch.transpose(y,-2,-1) @ grads)
        return grad_x, *grad_weights

class Embed(torch.autograd.Function):
    @staticmethod
    def forward(embedding,indx):
        return embedding[indx]
    
    @staticmethod
    def setup_context(ctx,inputs,outputs):
        ctx.save_for_backward(*inputs)
        
    @staticmethod
    def backward(ctx,grad_out):
        embedding,indx = ctx.saved_tensors
        dim = grad_out.size()[-1]
        grad_in = torch.zeros_like(embedding)
        return grad_in.index_add_(0,indx.reshape(-1),grad_out.reshape(-1,dim)), None
    
class Embedding(nn.Module):
    def __init__(self,tokens,dim):
        super().__init__()
        self.weights = nn.Parameter(torch.empty((tokens,dim)))
        nn.init.normal_(self.weights,mean=0,std=.02)
        self.embed = Embed.apply
    
    def forward(self,x):
        return self.embed(self.weights,x)
    
class RMSNormed(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.RMS = RMSNorm.apply
        self.gamma = nn.Parameter(torch.ones((1,dim)))
        
    def forward(self,x):
        return self.RMS(x,self.gamma)

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim,bias=True):
        super().__init__()
        self.enable_bias = bias
        self.weights = nn.Parameter(torch.empty((in_dim, out_dim)))
        nn.init.xavier_uniform_(self.weights)
        if self.enable_bias:
            self.bias = nn.Parameter(torch.zeros((1,out_dim)))

    def forward(self, x):
        y = x @ self.weights
        if self.enable_bias:
            return y + self.bias
        else:
            return y
        
class MaskedAttention(nn.Module):
    def __init__(self,heads,dim,dk,dv):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.heads = heads
        self.w_q = nn.Parameter(torch.randn((heads,dim,dk)))
        self.w_k = nn.Parameter(torch.randn((heads,dim,dk)))
        self.w_v = nn.Parameter(torch.randn((heads,dim,dv)))
        self.w_o = nn.Parameter(torch.randn((heads*dv,dim)))
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)
        nn.init.xavier_uniform_(self.w_o)
        self.QKV_projection = QKVProjection.apply
        self.softmax = SoftMax.apply
        
    def forward(self,x):
        Q,K,V = self.QKV_projection(x,self.w_q,self.w_k,self.w_v)
        batch_size = Q.size()[0]
        context = Q.size()[2]
        sqrt = math.sqrt(self.dk)
        scores = (Q @ torch.transpose(K, -2, -1)) / sqrt
        mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        QKV = self.softmax(scores) @ V
        QKV = QKV.transpose(1, 2).contiguous().view(batch_size, context, -1)
        return QKV @ self.w_o
    
class FeedForward(nn.Module):
    def __init__(self,dim,hidden):
        super().__init__()
        self.start = Linear(dim,hidden[0])
        self.linears = nn.ModuleList([Linear(hidden[i],hidden[i+1]) for i in range(len(hidden)-1)])
        self.end = Linear(hidden[-1],dim)
        self.smooth_relu = SmoothReLU.apply
    
    def forward(self,x):
        x = self.start(x)
        x = self.smooth_relu(x)
        for linear in self.linears:
            x = linear(x)
            x = self.smooth_relu(x)
        x = self.end(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,dim,heads,dk,dv,hidden):
        super().__init__()
        self.RMS1 = RMSNormed(dim)
        self.RMS2 = RMSNormed(dim)
        self.attention = MaskedAttention(heads,dim,dk,dv)
        self.feed_forward = FeedForward(dim,hidden)
        
    def forward(self,x):
        x = self.attention(self.RMS1(x)) + x
        x = self.feed_forward(self.RMS2(x)) + x
        return x
    
class Transformer(nn.Module):
    def __init__(self,tokens=TOKENS,dim=DIM,layers=LAYERS,heads=HEADS,dk=DK,dv=DV,hidden=HIDDEN):
        super().__init__()
        self.pe = PositionalEncoding()
        self.blocks = nn.ModuleList([TransformerBlock(dim,heads,dk,dv,hidden) for i in range(layers)])
        self.softmax = SoftMax.apply
        self.RMS = RMSNormed(dim)
        self.embed = Embedding(tokens,dim)
        self.decode = Linear(dim,tokens,bias=False)
               
        
    def forward(self,x):
        x = self.embed(x)
        x = self.pe(x)
        for block in self.blocks:
            x = block(x)
        x = self.RMS(x)
        return self.decode(x)
