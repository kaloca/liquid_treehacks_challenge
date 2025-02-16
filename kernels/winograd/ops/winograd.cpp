#include <torch/extension.h>
#include <torch/library.h>
#include <vector>
#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------
// Helper: Compute Vandermonde matrix
// Given a vector of points (length = rows), returns an (rows x cols) matrix
// with element (i, j) = points[i]^j (0-indexed, j from 0 to cols-1)
// ---------------------------------------------------------------------
static std::vector<float> compute_vandermonde(const std::vector<float> &points, int rows, int cols)
{
    std::vector<float> M(rows * cols, 0.0f);
    for (int i = 0; i < rows; i++)
    {
        float p = points[i];
        float power = 1.0f;
        for (int j = 0; j < cols; j++)
        {
            M[i * cols + j] = power;
            power *= p;
        }
    }
    return M;
}

// ---------------------------------------------------------------------
// Helper: Invert a square matrix (n x n) using Gauss-Jordan elimination.
// Matrix is stored in row-major order in vector<float> mat.
// On success, mat is replaced by its inverse.
// Returns true if successful, false if singular.
// ---------------------------------------------------------------------
static bool invert_matrix(std::vector<float> &mat, int n)
{
    std::vector<float> inv(n * n, 0.0f);
    for (int i = 0; i < n; i++)
    {
        inv[i * n + i] = 1.0f;
    }
    for (int i = 0; i < n; i++)
    {
        float pivot = mat[i * n + i];
        if (std::abs(pivot) < 1e-6f)
            return false;
        float inv_pivot = 1.0f / pivot;
        for (int j = 0; j < n; j++)
        {
            mat[i * n + j] *= inv_pivot;
            inv[i * n + j] *= inv_pivot;
        }
        for (int k = 0; k < n; k++)
        {
            if (k == i)
                continue;
            float factor = mat[k * n + i];
            for (int j = 0; j < n; j++)
            {
                mat[k * n + j] -= factor * mat[i * n + j];
                inv[k * n + j] -= factor * inv[i * n + j];
            }
        }
    }
    mat = inv;
    return true;
}

// ---------------------------------------------------------------------
// Helper: Matrix multiplication: C = A * B.
// A: (rowsA x colsA), B: (colsA x colsB); result C: (rowsA x colsB)
// ---------------------------------------------------------------------
static std::vector<float> matmul(const std::vector<float> &A, const std::vector<float> &B,
                                 int rowsA, int colsA, int colsB)
{
    std::vector<float> C(rowsA * colsB, 0.0f);
    for (int i = 0; i < rowsA; i++)
    {
        for (int j = 0; j < colsB; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < colsA; k++)
            {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }
    return C;
}

// ---------------------------------------------------------------------
// Helper: Matrix-vector multiplication: y = M * x.
// M: (rows x cols), x: (cols); result y: (rows)
// ---------------------------------------------------------------------
static void matvec(const std::vector<float> &M, const std::vector<float> &x,
                   std::vector<float> &y, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++)
        {
            sum += M[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

// ---------------------------------------------------------------------
// Choose output tile size (m) based on kernel size (K).
// This simple heuristic uses m=2 for small kernels, m=4 for moderate,
// and m=8 for larger kernels (K up to 32).
// ---------------------------------------------------------------------
static int choose_tile_size(int K)
{
    if (K <= 7)
        return 2;
    else if (K <= 16)
        return 4;
    else
        return 8;
}

// ---------------------------------------------------------------------
// Precompute Winograd transformation matrices for given kernel size K.
// Let r = K, m = chosen tile output size, and n = m + r - 1.
// We choose evaluation points for the input transform (p) as 0,1,..., n-1,
// for the filter transform (r_points) as 0,1,..., r-1, and for output
// interpolation (q) as 0,1,..., m-1.
//
// Then we compute:
//   G = V_p (n×r) * inv(V_r)        [filter transform: n x r]
//   B = inv(V_p_full)               [input transform: n x n], where V_p_full is Vandermonde for p (n×n)
//   A = V_q (m×n) * B              [output transform: m x n]
// ---------------------------------------------------------------------
struct WinogradTransforms
{
    int m;                // output tile size
    int r;                // kernel size (r = K)
    int n;                // tile size = m + r - 1
    std::vector<float> A; // m x n, row-major
    std::vector<float> B; // n x n, row-major
    std::vector<float> G; // n x r, row-major
};

static WinogradTransforms precompute_winograd(int K)
{
    WinogradTransforms wt;
    wt.r = K;
    wt.m = choose_tile_size(K);
    wt.n = wt.m + wt.r - 1;
    int n = wt.n, r = wt.r, m = wt.m;

    // p: evaluation points for input transform (size n)
    std::vector<float> p(n);
    for (int i = 0; i < n; i++)
    {
        p[i] = static_cast<float>(i);
    }
    // r_points: evaluation points for filter (size r)
    std::vector<float> r_points(r);
    for (int i = 0; i < r; i++)
    {
        r_points[i] = static_cast<float>(i);
    }
    // q: output interpolation points (size m)
    std::vector<float> q(m);
    for (int i = 0; i < m; i++)
    {
        q[i] = static_cast<float>(i);
    }

    // Compute Vandermonde matrices:
    // Vp_r: (n x r) for points p (using monomials 0..r-1)
    std::vector<float> Vp_r = compute_vandermonde(p, n, r);
    // Vr: (r x r) for points r_points.
    std::vector<float> Vr = compute_vandermonde(r_points, r, r);
    std::vector<float> Vr_inv = Vr;
    if (!invert_matrix(Vr_inv, r))
        throw std::runtime_error("Failed to invert Vr in Winograd precomputation.");
    // G = Vp_r * Vr_inv, size (n x r)
    wt.G = matmul(Vp_r, Vr_inv, n, r, r);

    // Compute full Vandermonde for p: Vp_full (n x n)
    std::vector<float> Vp_full = compute_vandermonde(p, n, n);
    std::vector<float> Vp_inv = Vp_full;
    if (!invert_matrix(Vp_inv, n))
        throw std::runtime_error("Failed to invert Vp in Winograd precomputation.");
    // Set B = inv(Vp_full)
    wt.B = Vp_inv;

    // Compute Vandermonde for q: Vq (m x n)
    std::vector<float> Vq = compute_vandermonde(q, m, n);
    // A = Vq * B, size (m x n)
    wt.A = matmul(Vq, wt.B, m, n, n);

    return wt;
}

// ---------------------------------------------------------------------
// Generic Winograd convolution using the precomputed transforms.
// Input: [B, C, N] (N = input length)
// Filter: [C, K] (per-channel filter)
// Output: [B, C, (N - K + 1)]
// For each channel, the filter is pretransformed: U = G * g.
// Then for each tile (of length n = m + K - 1) of the input,
// we compute: V = B * d, then M = U ∘ V, then y_tile = A * M.
// Leftover output positions are computed via a direct convolution fallback.
// ---------------------------------------------------------------------
static at::Tensor generic_winograd_conv(const at::Tensor &input, const at::Tensor &filter)
{
    TORCH_CHECK(input.dim() == 3, "Input must be [B, C, N]");
    TORCH_CHECK(filter.dim() == 2, "Filter must be [C, K]");
    TORCH_CHECK(input.size(1) == filter.size(0), "Channel mismatch between input and filter");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be float32");
    TORCH_CHECK(filter.scalar_type() == at::kFloat, "Filter must be float32");

    int64_t B = input.size(0), C = input.size(1), N = input.size(2);
    int64_t K = filter.size(1);
    TORCH_CHECK(K <= 32, "Kernel size K must be <= 32");
    int64_t out_len = N - K + 1;
    TORCH_CHECK(out_len > 0, "Input length must be >= kernel size");

    // Precompute the transforms.
    WinogradTransforms wt = precompute_winograd(K);
    int m = wt.m, n = wt.n;

    // Determine full tile count and leftover.
    int tile_count = out_len / m;
    int leftover = out_len % m;

    // Allocate output tensor [B, C, out_len].
    at::Tensor output = at::empty({B, C, out_len}, input.options());

    // Pretransform filters for each channel.
    // For each channel, compute U = G * g (g has length K, U has length n).
    std::vector<std::vector<float>> U_channel(C, std::vector<float>(n, 0.0f));
    const float *filter_ptr = filter.data_ptr<float>();
    for (int c = 0; c < C; c++)
    {
        std::vector<float> g(K, 0.0f);
        for (int k = 0; k < K; k++)
        {
            g[k] = filter_ptr[c * K + k];
        }
        std::vector<float> U(n, 0.0f);
        for (int i = 0; i < n; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < K; j++)
            {
                sum += wt.G[i * K + j] * g[j];
            }
            U[i] = sum;
        }
        U_channel[c] = std::move(U);
    }

    const float *input_ptr = input.data_ptr<float>();
    float *output_ptr = output.data_ptr<float>();

    // Process full tiles.
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < tile_count; t++)
        {
            int in_start = t * m; // input tile starts here (tile length = n)
            for (int c = 0; c < C; c++)
            {
                // Load input tile d of length n.
                std::vector<float> d(n, 0.0f);
                for (int i = 0; i < n; i++)
                {
                    d[i] = input_ptr[b * C * N + c * N + (in_start + i)];
                }
                // Compute V = B * d.
                std::vector<float> V(n, 0.0f);
                for (int i = 0; i < n; i++)
                {
                    float sum = 0.0f;
                    for (int j = 0; j < n; j++)
                    {
                        sum += wt.B[i * n + j] * d[j];
                    }
                    V[i] = sum;
                }
                // Elementwise multiply: M = U ∘ V.
                std::vector<float> M(n, 0.0f);
                const std::vector<float> &U = U_channel[c];
                for (int i = 0; i < n; i++)
                {
                    M[i] = U[i] * V[i];
                }
                // Compute output tile: y_tile = A * M, (length m).
                std::vector<float> y_tile(m, 0.0f);
                for (int i = 0; i < m; i++)
                {
                    float sum = 0.0f;
                    for (int j = 0; j < n; j++)
                    {
                        sum += wt.A[i * n + j] * M[j];
                    }
                    y_tile[i] = sum;
                }
                // Write y_tile into output at [b, c, t*m : t*m+m].
                int out_start = t * m;
                for (int i = 0; i < m; i++)
                {
                    output_ptr[b * C * out_len + c * out_len + (out_start + i)] = y_tile[i];
                }
            }
        }
        // Process leftover output positions with direct convolution.
        if (leftover > 0)
        {
            int out_start = tile_count * m;
            for (int c = 0; c < C; c++)
            {
                for (int pos = 0; pos < leftover; pos++)
                {
                    float sum = 0.0f;
                    int in_index = out_start + pos;
                    for (int k = 0; k < K; k++)
                    {
                        sum += input_ptr[b * C * N + c * N + (in_index + k)] *
                               filter_ptr[c * K + k];
                    }
                    output_ptr[b * C * out_len + c * out_len + (out_start + pos)] = sum;
                }
            }
        }
    }

    return output;
}

// ---------------------------------------------------------------------
// "Out" variant: write result into provided output tensor.
// ---------------------------------------------------------------------
static at::Tensor &generic_winograd_conv_out(const at::Tensor &input,
                                             const at::Tensor &filter,
                                             at::Tensor &out)
{
    at::Tensor result = generic_winograd_conv(input, filter);
    out.copy_(result);
    return out;
}

// ---------------------------------------------------------------------
// Meta (shape inference) variants.
// ---------------------------------------------------------------------
static at::Tensor generic_winograd_conv_meta(const at::Tensor &input, const at::Tensor &filter)
{
    TORCH_CHECK(input.dim() == 3, "Input must be [B, C, N]");
    TORCH_CHECK(filter.dim() == 2, "Filter must be [C, K]");
    int64_t B = input.size(0), C = input.size(1), N = input.size(2), K = filter.size(1);
    int64_t out_len = N - K + 1;
    return at::empty({B, C, out_len}, input.options().device(c10::kMeta));
}

static at::Tensor &generic_winograd_conv_meta_out(const at::Tensor &input,
                                                  const at::Tensor &filter,
                                                  at::Tensor &out)
{
    TORCH_CHECK(input.dim() == 3, "Input must be [B, C, N]");
    TORCH_CHECK(filter.dim() == 2, "Filter must be [C, K]");
    int64_t B = input.size(0), C = input.size(1), N = input.size(2), K = filter.size(1);
    int64_t out_len = N - K + 1;
    out = at::empty({B, C, out_len}, input.options().device(c10::kMeta));
    return out;
}

// ---------------------------------------------------------------------
// PyBind11 module registration.
// This allows: import winograd_generic && winograd_generic.generic_winograd_conv(...)
// ---------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generic_winograd_conv", &generic_winograd_conv, "Generic Winograd Convolution (float32)");
}

// ---------------------------------------------------------------------
// Torch Library registration so the op is accessible via torch.ops.
// ---------------------------------------------------------------------
at::Tensor generic_winograd_conv2(const at::Tensor &input, const at::Tensor &filter)
{
    // return generic_winograd_conv(input, filter);
    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t N = input.size(2);
    int64_t K = filter.size(1);
    int64_t out_len = N - K + 1;
    at::Device dev = input.is_meta() ? at::kCPU : input.device();
    at::Tensor out = at::empty({B, C, out_len}, input.options().device(dev));
    generic_winograd_conv_out(input, filter, out);

    return out;
}

TORCH_LIBRARY(multiwinograd, m)
{
    m.def("generic_winograd_conv(Tensor input, Tensor filter) -> Tensor", generic_winograd_conv2);
    m.def("generic_winograd_conv.out(Tensor input, Tensor filter, *, Tensor(a!) out) -> Tensor(a!)",
          generic_winograd_conv_out);
}

// ---------------------------------------------------------------------
// Executorch registration (if compiled with EXECUTORCH_LIBRARY defined).
// ---------------------------------------------------------------------
#ifdef EXECUTORCH_LIBRARY
EXECUTORCH_LIBRARY(multiwinograd, m)
{
    m.impl("generic_winograd_conv", generic_winograd_conv);
}
EXECUTORCH_LIBRARY(multiwinograd, m)
{
    m.impl("generic_winograd_conv.out", generic_winograd_conv_out);
}
#endif
