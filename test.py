from torch.autograd import gradcheck, gradgradcheck
import torch

class PositionalEncoding(torch.autograd.Function):
    @staticmethod
    def forward(x):
        context = x.size()[1]
        dim = x.size()[2]
        positions = torch.arange(context)[:, None]
        dimensions = torch.arange(dim)[None, :]
        angles = positions / (10000 ** (2 * (dimensions // 2) / dim))
        pe = torch.where(dimensions % 2 == 0, torch.sin(angles), torch.cos(angles))
        return x + pe
    
    @staticmethod
    def setup_context(ctx,inputs,outputs):
        pass
    
    @staticmethod
    def backward(ctx,grad_out):
        return grad_out
    
# Always test with float64 — float32 noise masks gradient errors
def verify_custom_function(func, *args):
    args_double = tuple(
        a.double().detach().requires_grad_(a.requires_grad) 
        if isinstance(a, torch.Tensor) and a.is_floating_point() else a
        for a in args
    )
    passed = gradcheck(func, args_double, eps=1e-6, atol=1e-4, raise_exception=True)
    print(f"First-order gradcheck: {'PASSED' if passed else 'FAILED'}")
    # Also check second-order gradients if you plan to use higher-order optimisers
    try:
        passed2 = gradgradcheck(func, args_double, eps=1e-6, atol=1e-4)
        print(f"Second-order gradcheck: {'PASSED' if passed2 else 'FAILED'}")
    except Exception as e:
        print(f"Second-order gradcheck not applicable: {e}")

x = torch.randn((2,3, 4), requires_grad=True)
verify_custom_function(PositionalEncoding.apply, x)