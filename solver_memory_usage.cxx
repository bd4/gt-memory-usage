#include <iostream>

#include <gtensor/gtensor.h>

#include <gt-solver/solver.h>

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

template <typename T, typename Solver> std::size_t test_solver_usage() {
  constexpr int n = 2 * 1024;
  constexpr int nrhs = 1;
  constexpr int nbatches = 256;
  constexpr int bw = 64;
  std::size_t mem_used = 0;

  mem_used = print_memusage("start    ", mem_used);

  auto h_A = make_test_matrix<T>(n, bw, nbatches, true);
  gt::gtensor<T, 3> h_rhs(gt::shape(n, nrhs, nbatches));
  gt::gtensor<T *, 1> h_Aptr(gt::shape(nbatches));

  mem_used = print_memusage("h alloc  ", mem_used);

  gt::gtensor_device<T, 3> d_rhs(h_rhs.shape());
  gt::gtensor_device<T, 3> d_result(h_rhs.shape());

  mem_used = print_memusage("vec alloc", mem_used);

  h_Aptr(0) = gt::raw_pointer_cast(h_A.data());
  for (int b = 1; b < nbatches; b++) {
    h_Aptr(b) = h_Aptr(0) + (n * n * b);
    for (int i = 0; i < n; i++) {
      for (int rhs = 0; rhs < nrhs; rhs++) {
        h_rhs(i, rhs, b) = T(1.0 + rhs / nrhs);
      }
    }
  }

  mem_used = print_memusage("h init   ", mem_used);

  gt::copy(h_rhs, d_rhs);

  mem_used = print_memusage("rhs copy ", mem_used);

  gt::blas::handle_t h;

  mem_used = print_memusage("handle   ", mem_used);

  Solver s(h, n, nbatches, nrhs, gt::raw_pointer_cast(h_Aptr.data()));

  mem_used = print_memusage("slv init ", mem_used);

  std::cout << typeid(Solver).name() << " devmem bytes "
            << static_cast<double>(s.get_device_memory_usage()) / (1024 * 1024)
            << " MB" << std::endl;

  s.solve(d_rhs.data().get(), d_result.data().get());

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
    mem_used = test_solver_usage<double, gt::solver::solver_dense<double>>();
    mem_used = print_memusage("main     ", mem_used);
  }
}
