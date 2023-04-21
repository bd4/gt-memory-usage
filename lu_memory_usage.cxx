#include <gt-blas/blas.h>
#include <gtensor/gtensor.h>

#ifdef GTENSOR_DEVICE_HIP
#include <rocblas.h>
#endif

#include <iostream>

void gpuMemGetInfo(size_t *free, size_t *total) {
#ifdef GTENSOR_DEVICE_CUDA
    gtGpuCheck(cudaMemGetInfo(free, total));
#elif defined(GTENSOR_DEVICE_HIP)
    gtGpuCheck(hipMemGetInfo(free, total));
#elif defined(GTENSOR_DEVICE_SYCL) && defined(GTENSOR_DEVICE_SYCL_L0)
    // Note: must set ZES_ENABLE_SYSMAN=1 in env for this to work
    gt::backend::sycl::mem_info(free, total);
#else
    // fallback so compiles and not divide by zero
    *total = 1;
    *free = 1;
#endif
}

size_t print_memusage(const char *prefix, std::size_t old_used = 0) {
    constexpr double GB = 1024 * 1024 * 1024;
    std::size_t free, total, used;
    long delta;
    gpuMemGetInfo(&free, &total);
    used = total - free;
    delta = used - old_used;
    std::cout << prefix << " " << used / GB << " / " << total / GB << ", delta "
              << delta / GB << std::endl;
    return used;
}

template <typename T>
auto make_test_matrix(int n, int bw, int batch_size, bool needs_pivot) {
    auto h_Adata = gt::zeros<T>({n, n, batch_size});
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            h_Adata(i, i, b) = T(bw + 1.0);
            // set upper / lower diags at bw diagonal
            for (int d = 1; d <= bw; d++) {
                if (i + d < n) {
                    h_Adata(i, i + d, b) = T(-1.0);
                    h_Adata(i + d, i, b) = T(-0.5);
                }
            }
        }
        if (needs_pivot) {
            h_Adata(0, 0, b) = T(n / 64.0);
        }
    }
    return h_Adata;
}

template <typename T>
std::size_t test_lu_usage() {
    constexpr int n = 2 * 1024;
    constexpr int nrhs = 1;
    constexpr int nbatches = 256;
    constexpr int bw = 64;
    std::size_t mem_used = 0;

    mem_used = print_memusage("start    ", mem_used);

    auto h_A = make_test_matrix<T>(n, bw, nbatches, true);
    auto d_A = gt::empty_device<T>(h_A.shape());

    gt::gtensor<T *, 1> h_Aptr(gt::shape(nbatches));
    gt::gtensor_device<T *, 1> d_Aptr(gt::shape(nbatches));

    gt::gtensor<T *, 1> h_Bptr(gt::shape(nbatches));
    gt::gtensor_device<T *, 1> d_Bptr(gt::shape(nbatches));

    gt::gtensor_device<gt::blas::index_t, 2> d_piv(gt::shape(n, nbatches));
    gt::gtensor_device<int, 1> d_info(gt::shape(nbatches));

    gt::gtensor<T, 3> h_rhs(gt::shape(n, nrhs, nbatches));
    gt::gtensor_device<T, 3> d_result(h_rhs.shape());

    mem_used = print_memusage("h/d alloc", mem_used);

    h_Aptr(0) = gt::raw_pointer_cast(d_A.data());
    h_Bptr(0) = gt::raw_pointer_cast(d_result.data());
    for (int b = 1; b < nbatches; b++) {
        h_Aptr(b) = h_Aptr(0) + (n * n * b);
        h_Bptr(b) = h_Bptr(0) + (n * nrhs * b);
        for (int i = 0; i < n; i++) {
            for (int rhs = 0; rhs < nrhs; rhs++) {
                h_rhs(i, rhs, b) = T(1.0 + rhs / nrhs);
            }
        }
    }

    mem_used = print_memusage("h init   ", mem_used);

    gt::copy(h_Aptr, d_Aptr);
    gt::copy(h_Bptr, d_Bptr);
    gt::copy(h_rhs, d_result);

    mem_used = print_memusage("d   copy ", mem_used);

    gt::blas::handle_t* h = new gt::blas::handle_t{};

    mem_used = print_memusage("handle   ", mem_used);

    gt::blas::getrf_batched(*h, n, d_Aptr.data().get(), n, d_piv.data().get(),
                            d_info.data().get(), nbatches);

    mem_used = print_memusage("slv init ", mem_used);

#ifdef GTENSOR_DEVICE_HIP
    rocblas_set_device_memory_size(h->get_backend_handle(), 0);
#else
    delete h;
#endif

    mem_used = print_memusage("delete h ", mem_used);

#ifndef GTENSOR_DEVICE_HIP
    h = new gt::blas::handle_t{};
#endif

    gt::blas::getrs_batched(*h, n, nrhs, d_Aptr.data().get(), n,
                            d_piv.data().get(), d_Bptr.data().get(), n,
                            nbatches);

    mem_used = print_memusage("slv solve", mem_used);

    gt::copy(d_result, h_rhs);

    mem_used = print_memusage("sln copy ", mem_used);

    return mem_used;
}

int main(int argc, char **argv) {
    constexpr int niter = 5;
    std::size_t mem_used;
    for (int i = 0; i < niter; i++) {
        std::cout << "========== " << i << " ===========" << std::endl;
        mem_used = test_lu_usage<double>();
        mem_used = print_memusage("main     ", mem_used);
    }
}
