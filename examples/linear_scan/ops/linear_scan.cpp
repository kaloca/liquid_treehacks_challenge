#include <torch/extension.h>
#include <ATen/Parallel.h>          
#include <c10/util/Exception.h>     
using torch::indexing::Ellipsis;
using torch::indexing::Slice;

// ---------------------------------------------------------------------
// 1) Compute kernel (CPU) - Original version
// Compute the state trajectory of x[t] = a[t] * x[t-1] + u[t] for x[0] = x0.
// u and a are tensors of shape u: [B, C, N], a: [B, C, N].
// ---------------------------------------------------------------------
at::Tensor fused_linear_scan(const at::Tensor& x0, const at::Tensor& a, const at::Tensor& u) {
    TORCH_CHECK(x0.scalar_type() == at::kFloat, "x0 must be float32");
    TORCH_CHECK(a.scalar_type() == at::kFloat,  "a must be float32");
    TORCH_CHECK(u.scalar_type() == at::kFloat,  "u must be float32");

    TORCH_CHECK(x0.dim() == 2, "Initial condition must be [B, C]");
    TORCH_CHECK(a.dim() == 3,  "Coeffiients must be [B, C, N]");
    TORCH_CHECK(u.dim() == 3,  "Input must be [B, C, N]");

    int64_t N = a.size(2);

    // Allocate output: x has shape [B, C, N]
    at::Tensor x = torch::zeros({a.size(0), a.size(1), N}, a.options());

    // x[..., 0] = a[..., 0] * x0 + u[..., 0]
    x.index_put_({Ellipsis, 0}, a.index({Ellipsis, 0}) * x0 + u.index({Ellipsis, 0}));

    // For t in [1..N-1], update: x[t] = a[t]*x[t-1] + u[t]
    for (int64_t t = 1; t < N; t++) {
        auto a_t = a.index({Ellipsis, t});
        auto u_t = u.index({Ellipsis, t});
        auto x_t_minus_1 = x.index({Ellipsis, t - 1});
        x.index_put_({Ellipsis, t}, a_t * x_t_minus_1 + u_t);
    }

    return x;
}

// ---------------------------------------------------------------------
// 2) Meta “shape-only” kernel (Optional: needed if you want a Meta backend).
// ---------------------------------------------------------------------
at::Tensor fused_linear_scan_meta(const at::Tensor& x0, const at::Tensor& a, const at::Tensor& u) {
    TORCH_CHECK(x0.dim() == 2, "Initial condition must be [B, C]");
    TORCH_CHECK(a.dim() == 3,  "Coeffiients must be [B, C, N]");
    TORCH_CHECK(u.dim() == 3,  "Input must be [B, C, N]");

    int64_t B = x0.size(0);
    int64_t C = x0.size(1);
    int64_t N = a.size(2);
    return at::empty({B, C, N}, x0.options().device(c10::kMeta));
}

// ---------------------------------------------------------------------
// 3) Optimized version of the fused linear scan.
//    This eliminates expensive indexing calls in the loop, using pointer arithmetic.
// ---------------------------------------------------------------------
at::Tensor fused_linear_scan2(
    const at::Tensor& x0,
    const at::Tensor& a,
    const at::Tensor& u
) {
    TORCH_CHECK(x0.scalar_type() == at::kFloat, "x0 must be float32");
    TORCH_CHECK(a.scalar_type() == at::kFloat,  "a must be float32");
    TORCH_CHECK(u.scalar_type() == at::kFloat,  "u must be float32");

    TORCH_CHECK(x0.dim() == 2, "x0 must be [B, C]");
    TORCH_CHECK(a.dim() == 3,  "a must be [B, C, N]");
    TORCH_CHECK(u.dim() == 3,  "u must be [B, C, N]");

    // Dimensions
    const auto B = a.size(0);
    const auto C = a.size(1);
    const auto N = a.size(2);

    // Make sure everything is contiguous
    auto x0_c = x0.contiguous();
    auto a_c  = a.contiguous();
    auto u_c  = u.contiguous();

    // Allocate output
    at::Tensor x = at::empty({B, C, N}, a.options());

    // Access raw pointers
    const float* x0_data = x0_c.data_ptr<float>();
    const float* a_data  = a_c.data_ptr<float>();
    const float* u_data  = u_c.data_ptr<float>();
    float*       x_data  = x.data_ptr<float>();

    // We will parallelize over B*C. Each row has length N.
    const int64_t BC = B * C;

    at::parallel_for(
        0,
        BC,
        /*grain_size=*/1,
        [&](int64_t start, int64_t end) {
            for (int64_t bc = start; bc < end; bc++) {
                // offset in the contiguous memory for the time dimension
                const int64_t offset_bc = bc * N;

                // Initial condition x[0] = a[0]*x0 + u[0]
                float x_prev = a_data[offset_bc] * x0_data[bc] + u_data[offset_bc];
                x_data[offset_bc] = x_prev;

                // Recurrence: x[t] = a[t]*x[t-1] + u[t]
                for (int64_t t = 1; t < N; t++) {
                    const float a_val = a_data[offset_bc + t];
                    const float u_val = u_data[offset_bc + t];
                    x_prev = a_val * x_prev + u_val;
                    x_data[offset_bc + t] = x_prev;
                }
            }
        }
    );

    return x;
}

// ---------------------------------------------------------------------
// 4) Single Pybind module exposing both versions for benchmarking.
//    This can be built via:
//       python setup.py install
//    or with a PyTorch C++ extension building script.
// ---------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_scan", &fused_linear_scan, "Fused Linear Scan (float32)");
    m.def("fused_linear_scan2", &fused_linear_scan2, "Fused Linear Scan Optimized (float32)");
}

// ---------------------------------------------------------------------
// 5) Torch library
// ---------------------------------------------------------------------
TORCH_LIBRARY(linear_scan, m) {
    m.def("fused_linear_scan(Tensor x0, Tensor a, Tensor u) -> Tensor");
    m.def("fused_linear_scan2(Tensor x0, Tensor a, Tensor u) -> Tensor");
}

// ---------------------------------------------------------------------
// 6) Torch library impl for CPU
// ---------------------------------------------------------------------
TORCH_LIBRARY_IMPL(linear_scan, CPU, m) {
    m.impl("fused_linear_scan", &fused_linear_scan);
    m.impl("fused_linear_scan2", &fused_linear_scan2);
}

// ---------------------------------------------------------------------
// 7) Torch library impl for Meta (shape-only kernel for the original).
//    If you want a shape-only kernel for the optimized version, you can add it similarly.
// ---------------------------------------------------------------------
TORCH_LIBRARY_IMPL(linear_scan, Meta, m) {
    m.impl("fused_linear_scan", fused_linear_scan_meta);
    m.impl("fused_linear_scan2", fused_linear_scan_meta);
}



