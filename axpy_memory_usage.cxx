#include <iostream>

#include <gtensor/gtensor.h>

#include <gt-blas/blas.h>

void gpuMemGetInfo(size_t* free, size_t* total)
{
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

void print_memusage(const char *prefix)
{
  std::size_t free, total;
  gpuMemGetInfo(&free, &total);
  std::cout << prefix << " " << free << " / " << total << std::endl;
}

int main(int argc, char **argv) {
  constexpr int N = 1024 * 1024 * 1024;

  print_memusage("start    ");

  gt::gtensor_device<double, 1> d_x(gt::shape(N));
  gt::gtensor_device<double, 1> d_y(gt::shape(N));

  print_memusage("x/y alloc");

  gt::gtensor<double, 1> h_x(gt::shape(N));
  gt::gtensor<double, 1> h_y(gt::shape(N));

  for (int i = 0; i < N; i++) {
    h_x(i) = static_cast<double>(i);
    h_y(i) = static_cast<double>(i);
  }

  gt::copy(h_x, d_x);
  gt::copy(h_y, d_y);

  print_memusage("copy     ");

  gt::blas::handle_t h;

  print_memusage("handle   ");

  gt::blas::axpy(h, 2.0, d_x, d_y);
  gt::copy(d_y, h_y);

  print_memusage("axpy     ");

  std::cout << "y[0]:   " << h_y(0) << "\n"
            << "y[N-1]: " << h_y(N - 1) << std::endl;
}
