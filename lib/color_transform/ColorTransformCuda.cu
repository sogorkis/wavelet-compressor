/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include <cutil_inline.h>

const int THREADS_PER_BLOCK = 512;  // Number of threads per block

__global__ void transformForwardCuda(float *bData, float *gData, float *rData, int length) {
    int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    if(i >= length) {
        return;
    }

    float r = rData[i];
    float g = gData[i];
    float b = bData[i];

    bData[i] = 0   + 0.2989 * r + 0.5866 * g + 0.1145 * b; // Y
    gData[i] = 128 - 0.1687 * r - 0.3312 * g + 0.5000 * b; // Cb
    rData[i] = 128 + 0.5000 * r - 0.4183 * g - 0.0816 * b; // Cr
}

__global__ void transformReverseCuda(float *bData, float *gData, float *rData, int length) {
    int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    if(i >= length) {
        return;
    }

    float y  = bData[i];
    float cb = gData[i] - 128;
    float cr = rData[i] - 128;

    bData[i] = y + 1.7650 * cb;                  // B
    gData[i] = y - 0.3456 * cb - 0.7145 * cr;    // G
    rData[i] = y + 1.4022 * cr;                  // R
}
