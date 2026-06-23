#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define SIZE 32768
#define BLOCKSIZE 1024

__global__ void layer_norm(float *in, float *out) {
    // per-warp vals
    __shared__ int count_arr[BLOCKSIZE / 32];
    __shared__ float mean_arr[BLOCKSIZE / 32];
    __shared__ float M2_arr[BLOCKSIZE / 32];

    // set thread id
    int tid = threadIdx.x;

    // values
    int count = 0;
    float mean = 0;
    float M2 = 0;

    // per-thread iteration
    for (int i = tid; i < SIZE; i += blockDim.x) {
        float x = in[i];

        // welford's 
        count += 1;
        float delta = x - mean; 
        mean += delta / count; // updates mean incrementally
        float delta2 = x - mean; // updated delta
        M2 += delta * delta2; // squared deviations, update previous
    }

    // per-warp merge
    for (int offset = 16; offset > 0; offset /= 2) {
        // grab next vals
        int count_next = __shfl_down_sync(0xffffffff, count, offset, 32);
        float mean_next = __shfl_down_sync(0xffffffff, mean, offset, 32);
        float M2_next = __shfl_down_sync(0xffffffff, M2, offset, 32);

        int count_curr = count;
        float mean_curr = mean;
        float M2_curr = M2;

        count = count_curr + count_next; // total count
        float delta = mean_next - mean_curr; // delta is diff between means
        mean = mean_curr + delta * ((float)count_next / count); // update mean - nudges towards B
        M2 = (M2_curr + M2_next) + delta * delta * ((float)count_curr * count_next / count); // squared deviations: spread within each group + spread between groups
    }

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // fill smem: only lane 0 has the full values of each warp
    if (lane_id == 0) {
        count_arr[warp_id] = count;
        mean_arr[warp_id] = mean;
        M2_arr[warp_id] = M2;
    }

    __syncthreads();

    int num_warps = blockDim.x / 32;

    // cross warp merge - only warp 0 does it
    if (warp_id == 0) {
        count = (lane_id < num_warps) ? count_arr[lane_id] : 0;
        mean = (lane_id < num_warps) ? mean_arr[lane_id] : 0;
        M2 = (lane_id < num_warps) ? M2_arr[lane_id] : 0;

        // merge - same as per warp
        for (int offset = 16; offset > 0; offset /= 2) {
            // grab next vals
            int count_next = __shfl_down_sync(0xffffffff, count, offset, 32);
            float mean_next = __shfl_down_sync(0xffffffff, mean, offset, 32);
            float M2_next = __shfl_down_sync(0xffffffff, M2, offset, 32);

            int count_curr = count;
            float mean_curr = mean;
            float M2_curr = M2;

            count = count_curr + count_next; // total count
            float delta = mean_next - mean_curr; // delta is diff between means
            mean = mean_curr + delta * ((float)count_next / count); // update mean - nudges towards B
            M2 = (M2_curr + M2_next) + delta * delta * ((float)count_curr * count_next / count); // squared deviations: spread within each group + spread between groups
        }
    }

    // write final vals - lane 0 has them
    if (tid == 0) {
        count_arr[0] = count;
        mean_arr[0] = mean;
        M2_arr[0] = M2;
    }

    __syncthreads();

    // final vals for every thread
    int count_final = count_arr[0];
    float mean_final = mean_arr[0];
    float final_var = M2_arr[0] / (float)count_final; // variance
    float inv_std = rsqrtf(final_var + 1e-5f); // inverse of standard deviation

    // write - strided loop since each thread does multiple
    for (int i = tid; i < SIZE; i += blockDim.x) {
        out[i] = (in[i] - mean_final) * inv_std;
    }
}