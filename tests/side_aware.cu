// ============================================================================
// Semianalysis Hackathon: "L2 Side Aware" (4++ hours extra to finish/cleanup/polish afterwards & turn it into something useful)
// (Got up to the point where it was reading data without transfers across the interconnect in NSight Compute during the hackathon)
// ============================================================================
// EXAMPLE COMMAND ('-r' for random input)
// ./QuickRunCUDA -i -p -t 1024 -A 1000000000 -0 1000000000 -T 100 -P 4.0 -U GB/s tests/side_aware.cu
// ============================================================================
// NVIDIA’s Blackwell B200 is 2 huge chips on 1 massive package which works like a single GPU from a software point of view.
// But physically half the cache/DRAM is on each side so 50% of accesses are for the other side with higher latency/power.
// This computes a reduction with each side of the GPU only reading elements in memory from its own side.
// Currently computes a FP32 absmax reduction but can easily be changed to any reduction
// Input size is passed with the "-0" argument as the number of floats/dwords (32 bits)
// ============================================================================
// Power results on GH200 96GiB (performance is roughly the same)
// Random data: 667W => 607W (-9.0%)
// Zeroed data: 474W => 442W (-6.8%)
// (I think it reduced power even more on Blackwell but I don't have one to retest with the latest version)
// ============================================================================
//#define FORCE_WRONG_SIDE
//#define FORCE_RANDOM_SIDE

//#define DEBUG_PRINTF_PER_PAGE
//#define DEBUG_PRINTF_FINAL_RESULT
//#define DEBUG_STORE_999_IN_A
#define DEBUG_PRINTF // used in init only so doesn't affect performance
#ifdef DEBUG_PRINTF
#define debugf(...) printf(__VA_ARGS__)
#else
#define debugf(...)
#endif

constexpr size_t CHUNK_SIZE = 4096; // 4KiB - determined empirically and now hardcoded, assert() vs actual hash in init()
constexpr size_t PAGE_SIZE = 2 * 1024 * 1024; // 2MiB (as documented by NVIDIA for low-level GPU virtual memory management)
constexpr int L2_SIDE_TEST_ITERATIONS = 25; // 10 seemed to have too large an error margin to be completely reliable
constexpr int FORCED_UNROLL = 8; // Always unroll the loop by this number of chunk (handle out-of-bounds afterwards)

constexpr int MAX_SM = 256; // oversized for future generations
constexpr int OFFSET_SM_SCRATCH = 512;
constexpr int OFFSET_PAGE_INFO = 1024;

constexpr int OFFSET_ZEROED_COUNTER = MAX_SM;
constexpr int OFFSET_AVERAGE_LATENCY = MAX_SM + 1;
constexpr int OFFSET_SIDE_HASH_MASK = MAX_SM + 2; // when using DETERMINE_HASH_DYNAMICALLY
constexpr int OFFSET_NUM_SM_SIDE0 = MAX_SM + 3;
constexpr int OFFSET_NUM_SM_SIDE1 = MAX_SM + 4;
constexpr int OFFSET_MIN_SM_PER_SIDE = MAX_SM + 5; // number of SMs on the side with the least SMs

#define DETERMINE_HASH_DYNAMICALLY
//constexpr int L2_HASH_BITS = 0x1EF000; // GB200 hash bits (determined during hackathon, see code below)
//constexpr int L2_HASH_BITS = 0x0AB000; // GH200 96GiB hash bits
//constexpr int L2_HASH_BITS = 0x0B3000; // H100 80GiB hash bits

// Key function that tests L2 latency for an address (on the current SM) using atomicAdd
// It saves the old value and restores it after the test automatically so any memory can be used
// (assuming no other thread is testing the latency of the same address at the same time!)
template<int iterations = L2_SIDE_TEST_ITERATIONS>
__device__ __forceinline__ int test_latency_l2(unsigned int* data, size_t offset) {
    unsigned int old_value = atomicExch(&data[offset], 0); // backup old value
    long long int start_clock = clock64();
    for (int i = 0; i < iterations; i++) {
        int value = atomicInc(&data[offset], 99999999);
        offset += (value > iterations*10) ? 1 : 0; // impossible condition so compiler plays along
    }
    int latency = clock64() - start_clock;
    data[offset] = old_value; // restore old value
    return latency;
}

// Initialization kernel that is run exactly once if QuickRunCUDA is called with "-i"
// It determines the average L2 latency for each SM *and* each 2MiB page
// Also determines which bits affect the side inside a given 2MiB page
extern "C" __global__  void init(float* A, unsigned int *side_info, float* output, int num_dwords, float dynamic_float, int unused_2) {
    if (threadIdx.x == 0) {
        assert(num_dwords > 0);
        size_t num_bytes = (size_t)num_dwords * 4;

        int smid;
        asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);

        // Try to spread out the atomicAdd operations which reduces contention (-> comparable to later latency tests)
        // More importantly it allows us to safely backup & restore that memory's original value in test_latency_l2()
        int offset = smid;
        assert(offset * sizeof(int) < CHUNK_SIZE);

        __nanosleep((smid % 8) * 1000); // Spread accesses through time a bit to reduce contention as well
        int total_latency = test_latency_l2((unsigned int*)A, offset);
        side_info[smid] = total_latency;
        atomicAdd(&side_info[OFFSET_AVERAGE_LATENCY], total_latency);

        int num_done = atomicInc(&side_info[OFFSET_ZEROED_COUNTER], gridDim.x - 1);
        if (num_done == gridDim.x - 1) {
            // Only a single SM executing past this point (last one to finish, so ironically probably the slowest...)
            // We use the average latency as the threshold for near vs far (only works because they're very different)
            int average_latency = side_info[OFFSET_AVERAGE_LATENCY] / gridDim.x;
            debugf("Average latency across all SMs: %.1f (= threshold for far vs near)\n", (float)average_latency / (float)L2_SIDE_TEST_ITERATIONS);

            int side0_counter = 0, side1_counter = 0;
            for (int i = 0; i < gridDim.x; i++) {
                int latency = side_info[i];
                side_info[i] = (side_info[i] > average_latency) ? 1 : 0; // Compare to average latency
                if (side_info[i] == 0) {
                    side_info[i] |= (side0_counter++) << 1;
                } else {
                    side_info[i] |= (side1_counter++) << 1;
                }
                debugf("[SM %3d] Average latency %.1f ===> SIDE[IDX]: %d [%d]\n",
                       i, (float)latency / (float)L2_SIDE_TEST_ITERATIONS, side_info[i], side_info[i] >> 1);
            }
            side_info[OFFSET_AVERAGE_LATENCY] = gridDim.x;

            // Store the total count of SMs on each side
            side_info[OFFSET_NUM_SM_SIDE0] = side0_counter;
            side_info[OFFSET_NUM_SM_SIDE1] = side1_counter;
            side_info[OFFSET_MIN_SM_PER_SIDE] = min(side0_counter, side1_counter); // Paired SMs we can actually use
            debugf("num_side_0: %d / num_side_1: %d\n", side0_counter, side1_counter);

            // Check memory address is 2MiB aligned (currently don't support unaligned input address)
            unsigned long long int addr_intptr = reinterpret_cast<unsigned long long int>(A);
            if (addr_intptr % (2 * 1024 * 1024) != 0) {
                debugf("ERROR: Data array is not 2MiB aligned, cannot easily determine hash or pages\n");
                assert(false);
                return;
            }

            #ifdef DETERMINE_HASH_DYNAMICALLY
            int base_side = side_info[smid] & 1;
            int check_start_bit = 4;
            int check_last_bit = log2f(PAGE_SIZE) - 1;
            int toggle_bits = 0;
            // Test each bit one by one and see if it affects the side
            for (int i = check_start_bit; i <= check_last_bit; i++) {
                int bitmask = 1 << i;

                int offset = bitmask / sizeof(int);
                int total_latency = test_latency_l2((unsigned int*)A, offset);
                int offset_side = (total_latency > average_latency) ? 1 : 0;
                if (offset_side != base_side) {
                    toggle_bits |= bitmask;
                }
            }
            side_info[OFFSET_SIDE_HASH_MASK] = toggle_bits;
            // Print every individual bit that was toggled
            debugf("======================================================================================\n");
            debugf("Part of address used for side selection inside a page: 0x%X (bits:", toggle_bits);
            for (int i = check_start_bit; i <= check_last_bit; i++) {
                int bitmask = 1 << i;
                if (toggle_bits & bitmask) {
                    debugf(" %d", i);
                }
            }
            debugf(")\n");
            // Make sure our hardcoded CHUNK_SIZE corresponds to the first bit set
            if (!(toggle_bits & CHUNK_SIZE) || (toggle_bits & (CHUNK_SIZE - 1))) {
                printf("\nERROR: CHUNK_SIZE %llu does not correspond to lowest hash bit\n\n", CHUNK_SIZE);
                assert(false);
            }
            #endif

            // Check memory addresses at 2MiB increments (page size) and determine if they're near or far
            unsigned int* page_info = side_info + OFFSET_PAGE_INFO; // Memory to store the results
            int num_double_chunks = num_bytes / (2 * CHUNK_SIZE);
            printf("num_bytes: %llu (num_double_chunks: %d)\n", num_bytes, num_double_chunks);
            int total_pages = (num_bytes + PAGE_SIZE - 1) / (PAGE_SIZE);

            debugf("======================================================================================\n");
            debugf("[SM %3d] Starting to test %d pages of %llu bytes each (TOTAL: %lld bytes)\n", smid, total_pages, PAGE_SIZE, (size_t)total_pages * (size_t)PAGE_SIZE);
            debugf("======================================================================================\n");
            #ifdef DEBUG_PRINTF_PER_PAGE
            printf("Average latency threshold for near vs far: %.1f\n", (float)average_latency / (float)L2_SIDE_TEST_ITERATIONS);
            #endif

            int num_near = 0;
            for (int page = 0; page < total_pages; page++) {
                // Calculate offset for this page (in int units)
                size_t page_offset = page * (PAGE_SIZE / sizeof(int));
                size_t page_latency = test_latency_l2((unsigned int*)A, page_offset);

                // Determine if this page is near or far based on average latency
                bool is_near = (page_latency <= average_latency);
                page_info[page] = is_near ? 0 : 1;
                num_near += is_near ? 1 : 0;
                page_info[page + total_pages] = page_latency; // for debugging only

                #ifdef DEBUG_PRINTF_PER_PAGE
                // Print every page for debugging
                printf("2MiB Page %4d: latency=%.1f, %s (0x%llx)\n",
                       page, (float)page_latency / (float)L2_SIDE_TEST_ITERATIONS,
                       is_near ? "NEAR" : "FAR ", addr_intptr + page * PAGE_SIZE);
                #endif
            }

            // Simple visualization of the page map
            debugf("Page map (0=near, 1=far): ");
            for (int page = 0; page < total_pages; page++) {
                debugf("%d", page_info[page]);
            }
            debugf("\nNear pages: %.1f%%\n", (float)num_near / (float)total_pages * 100.0f);

            #ifdef DEBUG_STORE_999_IN_A
            float* A_float = (float*)A;
            A_float[clock64() % (num_bytes/4)] = 1.0f;
            A_float[num_bytes/4 - 1] = 999.0f;
            #endif
        }
    } else {
        // Make sure the threadgroup doesn't exit early which might affect work distribution across SMs on some HW
        // (not sure if needed on Blackwell, better safe than sorry)
        __nanosleep(10000);
    }
}

// =====================================================================================================================
// Main kernel & its helper functions
// =====================================================================================================================
// This is the reduction operation done by this kernel (easily changed to any other reduction operation)
constexpr float reduction_oob = 0.0f;
__device__ inline float reduction_op(float a, float b) {
    return fmaxf(a, fabsf(b));
}
// These are warp and block reduction kernels that I previously wrote for llm.c
// For H100 absmax specifically, we could use "redux.sync.max.u32" but this is more flexible
using warp_reduction_func_t = float (*) (float);
using thread_reduction_func_t = float (*) (float, float);
template<thread_reduction_func_t thread_reduction>
__device__ inline float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = thread_reduction(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}
// requires all 32 threads in the warp to be active, but should work for any block size
template<warp_reduction_func_t warp_reduction>
__device__ inline float block_reduce(float val, bool final_sync=false, float out_of_bounds=reduction_oob) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    constexpr unsigned int WARP_SIZE = 32;
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warp_reduction(val);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val);

    if (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    return block_val;
}

// Main kernel that does a L2-side-aware reduction of the input data
// This is called 100 times if QuickRunCUDA is called with "-T 100"
// __launch_bounds__ is very important otherwise the FORCED_UNROLL loop is not properly unrolled
extern "C" __global__ __launch_bounds__(1024, 1, 1) void kernel(float *A, unsigned int* side_info, float* output, int num_dwords, float dynamic_float, int unused_2) {
    int smid;
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);

    // Get this SM's side (0 or 1) and its index within that side
    int sm_side = side_info[smid] & 1;
    int sm_side_index = side_info[smid] >> 1;
    int num_sm_per_side = side_info[OFFSET_MIN_SM_PER_SIDE];

    // Exit early if there's an unbalanced number of SMs on each side, and this is an "extra SM" on the larger side
    if (sm_side_index >= num_sm_per_side) {
        return;
    }

    // These are the (virtual memory address) bits used in the hash that determines the side of an address
    #ifdef DETERMINE_HASH_DYNAMICALLY
    int hash_mask = side_info[OFFSET_SIDE_HASH_MASK];
    #else
    int hash_mask = SIDE_HASH_BITS;
    #endif
    // Along with the physical address bits we cannot know and so we determine their impact empirically for page_side
    unsigned int* page_info = side_info + OFFSET_PAGE_INFO;

    // Each "group" is 256 threads processing 16 bytes each = 4096 bytes
    int num_groups = blockDim.x / 256;
    int group = threadIdx.x / 256;
    int group_tid = threadIdx.x % 256;
    int num_groups_per_side = num_sm_per_side * num_groups;
    int global_idx = (sm_side_index * num_groups) + group;

    // We support an arbitrary number of bytes with partial chunks and out-of-bounds checking
    size_t num_bytes = (size_t)num_dwords * 4;
    int num_double_chunks = num_bytes / (2 * CHUNK_SIZE);
    int multi_chunks = num_double_chunks / FORCED_UNROLL;

    float result = 0.0f;
    size_t i = global_idx;
    for (; i < multi_chunks; i += num_groups_per_side) {
        #pragma unroll
        for (int j = 0; j < FORCED_UNROLL; j++) {
            size_t offset = (i * FORCED_UNROLL + j) * 2 * CHUNK_SIZE;

            // Determine the side of the 1st 4KiB chunk in the 8KiB "double chunk"
            int page_side = page_info[offset / PAGE_SIZE];
            int lsb_side_bits = (offset & hash_mask);
            int lsb_side = (__popcll(lsb_side_bits) & 1);
            int side = lsb_side ^ page_side;

            #ifdef FORCE_WRONG_SIDE
            side ^= 1;
            #else
            #ifdef FORCE_RANDOM_SIDE
            side ^= page_side;
            #endif
            #endif

            if (side == sm_side) {
                // Switch to the 2nd 4KiB chunk in a 8KiB "double chunk" if the 1st 4KiB is on the other side
                // The paired SM from the other side will be responsible for the opposite 4KiB chunk
                // (since these two 4KiB chunks are always from opposite sides)
                offset += CHUNK_SIZE;
            }
            offset += group_tid * 16;

            unsigned long long addr = reinterpret_cast<unsigned long long>(A) + offset;
            float4 input_data = *((float4*)addr);
            for (int k = 0; k < 4; k++) {
                result = reduction_op(result, ((float*)&input_data)[k]);
            }
        }
    }

    // Process the remaining data that isn't a multiple of (2 * CHUNK_SIZE * FORCED_UNROLL)
    size_t byte_offset = (i * FORCED_UNROLL) * 2 * CHUNK_SIZE;
    for (; byte_offset < num_bytes; byte_offset += 2 * CHUNK_SIZE) {
        // Determine the side of the 1st 4KiB chunk in the 8KiB "double chunk"
        int page_side = page_info[byte_offset / PAGE_SIZE];
        int lsb_side_bits = (byte_offset & hash_mask);
        int lsb_side = (__popcll(lsb_side_bits) & 1);
        int side = lsb_side ^ page_side;

        size_t offset = byte_offset + group_tid * 16;
        if (side == sm_side) {
            // Switch to the 2nd 4KiB chunk in a 8KiB "double chunk" if the 1st 4KiB is on the other side
            // The paired SM from the other side will be responsible for the opposite 4KiB chunk
            // (since these two 4KiB chunks are always from opposite sides)
            offset += CHUNK_SIZE;
        }

        if (offset + sizeof(float4) <= num_bytes) {
            unsigned long long addr = reinterpret_cast<unsigned long long>(A) + offset;
            float4 input_data = *((float4*)addr);
            for (int k = 0; k < 4; k++) {
                result = reduction_op(result, ((float*)&input_data)[k]);
            }
        }
    }

    // Process the final partial float4 (when the number of dwords is not a multiple of 4)
    if (blockIdx.x == 0 && threadIdx.x == 0 && (num_bytes % sizeof(float4) != 0)) {
        size_t offset = num_bytes - (num_bytes % sizeof(float4));
        for (; offset < num_bytes; offset += sizeof(float)) {
            result = reduction_op(result, A[offset / sizeof(float)]);
        }
    }

    // Reduction across the entire block
    result = block_reduce<warp_reduce<reduction_op>>(result);

    // The following is a highly optimized *fully deterministic* global reduction (for any reduction operation)
    // The final reduction operations are computed on the last block to finish (with a single warp)
    // It doesn't use any atomics except atomicInc to determine which block/SM is last to finish
    int done = 0;
    float* scratch = (float*)(side_info + OFFSET_SM_SCRATCH);
    if (threadIdx.x == 0) {
        int num_done = atomicInc(&side_info[OFFSET_ZEROED_COUNTER], 2*num_sm_per_side-1);
        scratch[sm_side_index + sm_side * num_sm_per_side] = result;
        done = (num_done == 2*num_sm_per_side-1);
    }

    int final_warp = __any_sync(0xFFFFFFFF, done);
    if (final_warp) {
        float final_result = 0.0f;
        #pragma unroll
        for (int i = 0; i < MAX_SM / 2; i += 32) {
            float value_side0 = scratch[i + threadIdx.x];
            float value_side1 = scratch[i + threadIdx.x + num_sm_per_side];
            if (i + threadIdx.x < num_sm_per_side) { // unrolling + predication --> fast!
                final_result = reduction_op(reduction_op(final_result, value_side0), value_side1);
            }
        }
        final_result = warp_reduce<reduction_op>(final_result);
        output[0] = final_result;

        #ifdef DEBUG_PRINTF_FINAL_RESULT
        printf("Final result: %f\n", final_result);
        #endif
    }
}
