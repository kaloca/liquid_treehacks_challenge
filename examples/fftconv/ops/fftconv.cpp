#include <torch/extension.h>
#include <torch/fft.h>

// ---------------------------------------------------------------------
// 1) Compute kernel (CPU)
// ---------------------------------------------------------------------
at::Tensor fused_fftconv(const at::Tensor& input, const at::Tensor& filter) {
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be Float32");
    TORCH_CHECK(filter.scalar_type() == at::kFloat, "Filter must be Float32");
    TORCH_CHECK(input.dim() == 3, "Input must be [B, C, N]");
    TORCH_CHECK(filter.dim() == 2, "Filter must be [C, K]");
    TORCH_CHECK(input.size(1) == filter.size(0), "Channel mismatch");
    int64_t N = input.size(2);
    int64_t K = filter.size(1);
    int64_t L = N + K - 1;
    int64_t N_fft = 1;
    while (N_fft < L) { N_fft <<= 1; }

    at::Tensor X = input.contiguous();
    at::Tensor H = filter.contiguous();
    at::Tensor X_fft = torch::fft::rfft(X, N_fft, /*dim*/ -1);
    at::Tensor H_fft = torch::fft::rfft(H, N_fft, /*dim*/ -1);

    H_fft = H_fft.unsqueeze(0); 
    X_fft.mul_(H_fft);

    at::Tensor conv_full = torch::fft::irfft(X_fft, N_fft, /*dim*/ -1);
    // slice back to original length
    return conv_full.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, N)});
}

// ---------------------------------------------------------------------
// 2) Meta "shape-only" kernel
// ---------------------------------------------------------------------
at::Tensor fused_fftconv_meta(const at::Tensor& input, const at::Tensor& filter) {
    // same shape checks (ideally keep them consistent)
    TORCH_CHECK(input.dim() == 3, "Input must be [B, C, N]");
    TORCH_CHECK(filter.dim() == 2, "Filter must be [C, K]");
    TORCH_CHECK(input.size(1) == filter.size(0), "Channel mismatch");
    // Output shape is [B, C, N]
    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t N = input.size(2);
    return at::empty({B, C, N}, input.options().device(c10::kMeta));
}

// ---------------------------------------------------------------------
// 3) Pybind module, so we can do "import fused_fftconv" directly
// ---------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_fftconv", &fused_fftconv, "Fused FFT Convolution (Float32)");
}

// ---------------------------------------------------------------------
// 4) Torch Library registration
// ---------------------------------------------------------------------
TORCH_LIBRARY(fftconv, m) {
    // Declare the schema
    m.def("fused_fftconv(Tensor input, Tensor filter) -> Tensor");
}

// CPU Implementation
TORCH_LIBRARY_IMPL(fftconv, CPU, m) {
    m.impl("fused_fftconv", &fused_fftconv);
}

// Meta Implementation (for shape inference on fake tensors)
TORCH_LIBRARY_IMPL(fftconv, Meta, m) {
    m.impl("fused_fftconv", fused_fftconv_meta);
}

#ifdef EXECUTORCH_LIBRARY
// ---------------------------------------------------------------------
// 5) Executorch Implementation
// ---------------------------------------------------------------------    
EXECUTORCH_LIBRARY(fftconv, m) {
    m.impl("fused_fftconv", fused_fftconv);
}
#endif // EXECUTORCH_LIBRARY
