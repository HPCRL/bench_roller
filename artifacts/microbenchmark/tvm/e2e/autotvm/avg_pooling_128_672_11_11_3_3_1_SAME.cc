//10164_1_1_1024_1_1
//avg_128_672_11_11_3_1_SAME
//dim3 grid(10164, 1, 1);
//dim3 block(1024, 1, 1);

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(1024) default_function_kernel0(float* __restrict__ data, float* __restrict__ tensor) {
  float tensor1[1];
  tensor1[(0)] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      tensor1[(0)] = (tensor1[(0)] + (((((1 <= (((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 121) / 11) + rv0)) && ((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 121) / 11) + rv0) < 12)) && (1 <= (rv1 + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 11)))) && ((rv1 + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 11)) < 12)) ? data[((((((((int)blockIdx.x) * 1024) + (rv0 * 11)) + ((int)threadIdx.x)) + rv1) - 12))] : 0.000000e+00f));
    }
  }
  tensor[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = (tensor1[(0)] * 1.111111e-01f);
}

