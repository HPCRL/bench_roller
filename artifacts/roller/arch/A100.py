from .Arch import *

class A100(Arch):
    # Compute Capability 8.0
    def __init__(self, para_opt=True):
        super().__init__()
        self.num_level = 2
        # DRAM: memory level 0
        # SMEM: memory level 1
        # REG: memory level 2
        # Bandwidth (unit: GB/s)
        self.bandwidth = [1555, 1555]  # HBM2 memory bandwidth is approximately 1.6 TB/s
        # Compute throughput (unit: GFLOPS)
        self.peak_flops = 19500  # Single-precision floating-point performance is approximately 19.5 TFLOPS
        self.peak_tc_flops = 312000  # Tensor Core performance is approximately 312 TFLOPS
        self.limit = []
        self.reg_cap = [65536, 32]  # Number of registers per SM and bits per register
        self.smem_cap = [164 * 1024]  # Shared memory capacity per SM, unit: bytes
        self.compute_max_core = [108, 108 * 64]  # Number of SMs and FP32 cores per SM
        self.mem_max_core = [108, 108 * 64]
        self.para_opt = para_opt
        self.warp_size = 32
        self.compute_sm_partition = [108, 4]
        self.smem_sm_partition = [108, 4]
        self.compute_block_schedule_way = ['warp', 'active block']
        self.smem_block_schedule_way = ['warp', 'active block']
        self.transaction_size = [32, 128]  # Unit: bytes
        self.glbmem_sm_partition = [108, 32]  # Number of warps per SM to achieve peak global memory throughput
        self.smem_bank_size = 4
        self.bank_number = 32
        self.compute_capability = 'compute_80'  # Compute capability of A100 is 8.0
        # Parameters for estimating active blocks
        self.max_active_blocks = 32
        self.max_smem_usage = 164 * 1024 - 1  # Maximum shared memory usage
        self.max_threads_per_sm = 2048  # Maximum number of threads per SM
