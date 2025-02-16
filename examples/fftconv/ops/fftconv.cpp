#include <torch/extension.h>
#include <torch/fft.h>
#include <torch/library.h>
// ---------------------------------------------------------------------
// 1) Compute kernel (CPU)
// ---------------------------------------------------------------------
at::Tensor fused_fftconv(const at::Tensor &input, const at::Tensor &filter)
{
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be Float32");
    TORCH_CHECK(filter.scalar_type() == at::kFloat, "Filter must be Float32");
    TORCH_CHECK(input.dim() == 3, "Input must be [B, C, N]");
    TORCH_CHECK(filter.dim() == 2, "Filter must be [C, K]");
    TORCH_CHECK(input.size(1) == filter.size(0), "Channel mismatch");
    int64_t N = input.size(2);
    int64_t K = filter.size(1);
    int64_t L = N + K - 1;
    int64_t N_fft = 1;
    while (N_fft < L)
    {
        N_fft <<= 1;
    }

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
at::Tensor fused_fftconv_meta(const at::Tensor &input, const at::Tensor &filter)
{
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
// 1) Compute kernel (CPU) out
// ---------------------------------------------------------------------
at::Tensor &fused_fftconv_out(const at::Tensor &input, const at::Tensor &filter, at::Tensor &out)
{
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be Float32");
    TORCH_CHECK(filter.scalar_type() == at::kFloat, "Filter must be Float32");
    TORCH_CHECK(input.dim() == 3, "Input must be [B, C, N]");
    TORCH_CHECK(filter.dim() == 2, "Filter must be [C, K]");
    TORCH_CHECK(input.size(1) == filter.size(0), "Channel mismatch");

    // Get dimensions.
    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t N = input.size(2);
    int64_t K = filter.size(1);

    // Reshape filter to shape [C, 1, K] so it can be used as conv1d weight.
    at::Tensor weight = filter.view({C, 1, K});

    // Pad the input on the left with (K-1) zeros.
    // The padding format for constant_pad_nd is {pad_left, pad_right}.
    at::Tensor padded = torch::constant_pad_nd(input, {K - 1, 0});

    // Perform a 1D convolution:
    // - stride = 1, no additional padding (we already padded manually),
    // - groups = C (so that each channel is convolved separately).
    out = torch::conv1d(
        padded,
        weight,
        /*bias=*/std::nullopt,
        at::IntArrayRef({1}),
        at::IntArrayRef({0}),
        at::IntArrayRef({1}),
        C);
    // int64_t N = input.size(2);
    // int64_t K = filter.size(1);
    // int64_t L = N + K - 1;
    // int64_t N_fft = 1;
    // while (N_fft < L)
    // {
    //     N_fft <<= 1;
    // }

    // at::Tensor X = input.contiguous();
    // at::Tensor H = filter.contiguous();
    // at::Tensor X_fft = torch::fft::rfft(X, N_fft, /*dim*/ -1);
    // at::Tensor H_fft = torch::fft::rfft(H, N_fft, /*dim*/ -1);

    // H_fft = H_fft.unsqueeze(0);
    // X_fft.mul_(H_fft);

    // out = torch::fft::irfft(X_fft, N_fft, /*dim*/ -1);
    // // slice back to original length
    // out = out.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, N)});

    return out;
}

// ---------------------------------------------------------------------
// 2) Meta "shape-only" kernel out
// ---------------------------------------------------------------------
at::Tensor &fused_fftconv_meta_out(const at::Tensor &input, const at::Tensor &filter, at::Tensor &out)
{
    // same shape checks (ideally keep them consistent)
    TORCH_CHECK(input.dim() == 3, "Input must be [B, C, N]");
    TORCH_CHECK(filter.dim() == 2, "Filter must be [C, K]");
    TORCH_CHECK(input.size(1) == filter.size(0), "Channel mismatch");
    // Output shape is [B, C, N]
    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t N = input.size(2);

    out = at::empty({B, C, N}, input.options().device(c10::kMeta));
    return out;
}

// ---------------------------------------------------------------------
// 3) Pybind module, so we can do "import fused_fftconv" directly
// ---------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fused_fftconv", &fused_fftconv, "Fused FFT Convolution (Float32)");
}

// ---------------------------------------------------------------------
// 4) Torch Library registration
// ---------------------------------------------------------------------
// Standard fused_fftconv function
at::Tensor fused_fftconv2(const at::Tensor &input, const at::Tensor &filter)
{
    at::Tensor out = at::empty({input.size(0), input.size(1), input.size(2)}, input.options());
    fused_fftconv_out(input, filter, out);
    return out;
}

// Standard API to register ops into PyTorch
TORCH_LIBRARY(fftconv, m)
{
    m.def("fused_fftconv(Tensor input, Tensor filter) -> Tensor", fused_fftconv2);
    m.def("fused_fftconv.out(Tensor input, Tensor filter, *, Tensor(a!) out) -> Tensor(a!)",
          fused_fftconv_out);
}

// CPU Implementation
// TORCH_LIBRARY_IMPL(fftconv, CPU, m)
// {
//     m.impl("fused_fftconv", &fused_fftconv);
// }

// // Meta Implementation (for shape inference on fake tensors)
// TORCH_LIBRARY_IMPL(fftconv, Meta, m)
// {
//     m.impl("fused_fftconv", fused_fftconv_meta);
// }

#ifdef EXECUTORCH_LIBRARY
// ---------------------------------------------------------------------
// 5) Executorch Implementation
// ---------------------------------------------------------------------
EXECUTORCH_LIBRARY(fftconv, m)
{
    m.impl("fused_fftconv", fused_fftconv);
}
EXECUTORCH_LIBRARY(fftconv, m)
{
    m.impl("fused_fftconv.out", fused_fftconv_out);
}
// EXECUTORCH_LIBRARY(fftconv, "fused_fftconv.out", fused_fftconv_out);

#endif // EXECUTORCH_LIBRARY