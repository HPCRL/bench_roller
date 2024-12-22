from .Arch import *

class RTX4090(Arch):
    # Compute Capability 8.9
    def __init__(self, para_opt=True):
        super().__init__()
        self.num_level = 2
        # DRAM: memory level 0
        # SMEM: memory level 1
        # REG: memory level 2
        # Bandwidth (unit: GB/s)
        self.bandwidth = [1000, 19328]  # Updated DRAM bandwidth
        # Compute throughput (unit: GFLOPS)
        self.peak_flops = 82580  # Updated single-precision floating-point performance
        self.peak_tc_flops = 1321000  # Updated Tensor Core performance
        self.limit = []
        self.reg_cap = [32768, 96]  # Number of registers per SM and bits per register
        self.smem_cap = [65536]  # Updated shared memory capacity, unit: bytes
        self.compute_max_core = [128, 128 * 4 * 32]  # Number of SMs and cores per SM
        self.mem_max_core = [128, 128 * 4 * 32]
        self.para_opt = para_opt
        self.warp_size = 32
        self.compute_sm_partition = [128, 4]
        self.smem_sm_partition = [128, 4]
        self.compute_block_schedule_way = ['warp', 'active block']
        self.smem_block_schedule_way = ['warp', 'active block']
        self.transaction_size = [32, 128]  # Unit: bytes
        self.glbmem_sm_partition = [128, 32]  # Number of warps per SM to achieve peak global memory throughput
        self.smem_bank_size = 4
        self.bank_number = 32
        self.compute_capability = 'compute_89'  # Compute capability of RTX 4090
        # Parameters for estimating active blocks
        self.max_active_blocks = 32
        self.max_smem_usage = 64 * 1024 - 1  # Maximum shared memory usage
        self.max_threads_per_sm = 1024  # Maximum number of threads per SM
