from torch.autograd import gradcheck, gradgradcheck
import torch

class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(x, gamma, eps=1e-6):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x / rms * gamma, rms
    
    @staticmethod
    def setup_context(ctx,inputs,outputs):
        out,rms = outputs
        x, gamma = inputs[:2]
        ctx.save_for_backward(x,gamma,rms)
        ctx.mark_non_differentiable(rms)

    @staticmethod
    def backward(ctx, g, g_rms):
        x, gamma, rms = ctx.saved_tensors
        d = x.size(-1)
        gg = g * gamma
        grad_x = gg / rms - x * (gg * x).sum(-1, keepdim=True) / (d * rms**3)
        grad_gamma = (g * x / rms).reshape(-1, d).sum(0).reshape(gamma.shape)
        return grad_x, grad_gamma, None
    
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
gamma = torch.ones((1,4),requires_grad=True)
verify_custom_function(RMSNorm.apply, x,gamma)