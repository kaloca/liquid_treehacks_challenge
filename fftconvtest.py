import torch
import torch.nn as nn
import fused_fftconv  # Import your compiled extension

class FFTConvModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, filter):
        return fused_fftconv.fused_fftconv(x, filter)

# Example input
B, C, N, K = 4, 8, 128, 32  # Adjust as needed
example_input = torch.randn(B, C, N, dtype=torch.float32)
example_filter = torch.randn(C, K, dtype=torch.float32)

# Instantiate module
module = FFTConvModule()