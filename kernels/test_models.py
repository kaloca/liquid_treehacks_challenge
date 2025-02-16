import torch
import torch.nn as nn

import fused_fftconv  # Import your compiled extension
import generic_winograd_conv

class FFTConvModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, filter):
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        torch.ops.fftconv.fused_fftconv.out(x, filter, out=out)  # Call the out= variant
        return out
    

@torch._dynamo.disable()
def opaque_winograd_conv(x, filt):
    B, C, N = x.shape
    K = filt.shape[1]
    out_len = N - K + 1
    # Allocate output on the same device/dtype as x (force real allocation)
    out = torch.empty((B, C, out_len), dtype=x.dtype, device=x.device)
    # Call the custom op using its "out" variant.
    torch.ops.multiwinograd.generic_winograd_conv.out(x, filt, out=out)
    return out

class WinogradConvModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, filter):
        return opaque_winograd_conv(x, filter)