/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "YCbCrColorTransformCuda.h"
#include "ColorTransformCuda.cu"

void YCbCrColorTransformCuda::transformForwardDevice(float **input, int length) {
    int blockNum = 1;
    if(length > THREADS_PER_BLOCK) {
        blockNum = length / THREADS_PER_BLOCK;
        blockNum += length % THREADS_PER_BLOCK != 0 ? 1 : 0;
    }

    transformForwardCuda<<<blockNum, THREADS_PER_BLOCK>>>(input[0], input[1], input[2], length);
}

void YCbCrColorTransformCuda::transformReverseDevice(float **input, int length) {
    int blockNum = 1;
    if(length > THREADS_PER_BLOCK) {
        blockNum = length / THREADS_PER_BLOCK;
        blockNum += length % THREADS_PER_BLOCK != 0 ? 1 : 0;
    }

    transformReverseCuda<<<blockNum, THREADS_PER_BLOCK>>>(input[0], input[1], input[2], length);
}
