/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "WaveletTransformCudaImpl.h"
#include "CudaWaveletTransform.cu"
#include "util/WaveletCompressorUtil.h"
#include "util/CudaEventTimer.h"

#include <cutil_inline.h>

void WaveletTransformCudaImpl::forward1D(const float *input, float *output, int length, int levels, Wavelet *wavelet) {
    float * deviceInput;
    float * deviceOutput;

    // Allocate device memory and copy input data
    cutilSafeCall(cudaMalloc((void**)&deviceInput, length * sizeof(float)));
    cutilSafeCall(cudaMalloc((void**)&deviceOutput, length * sizeof(float)));
    cutilSafeCall(cudaMemcpy(deviceInput, input, length * sizeof(float), cudaMemcpyHostToDevice));

    // Perform forward transform on device
    forwardDeviceMemory1D(deviceInput, deviceOutput, length, levels, wavelet);

    // Copy output data and free device memory
    cutilSafeCall(cudaMemcpy(output, deviceOutput, length * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}

void WaveletTransformCudaImpl::reverse1D(const float *input, float *output, int length, int levels, Wavelet *wavelet) {
    float * deviceInput;
    float * deviceOutput;

    // Allocate device memory and copy input data
    cutilSafeCall(cudaMalloc((void**)&deviceInput, length * sizeof(float)));
    cutilSafeCall(cudaMalloc((void**)&deviceOutput, length * sizeof(float)));
    cutilSafeCall(cudaMemcpy(deviceInput, input, length * sizeof(float), cudaMemcpyHostToDevice));

    // Perform reverse transform on device
    reverseDeviceMemory1D(deviceInput, deviceOutput, length, levels, wavelet);

    // Copy output data and free device memory
    cutilSafeCall(cudaMemcpy(output, deviceOutput, length * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}

void WaveletTransformCudaImpl::forward2D(const float *input, float *output, int width, int height, int levels, Wavelet *wavelet) {
    float * deviceInput;
    float * deviceOutput;
    int length = width * height;

    // Allocate device memory and copy input data
    cutilSafeCall(cudaMalloc((void**)&deviceInput, length * sizeof(float)));
    cutilSafeCall(cudaMalloc((void**)&deviceOutput, length * sizeof(float)));
    cutilSafeCall(cudaMemcpy(deviceOutput, input, length * sizeof(float), cudaMemcpyHostToDevice));

    // Perform forward transform on device
    forwardDeviceMemory2D(deviceInput, deviceOutput, width, height, levels, wavelet);

    // Copy output data and free device memory
    cutilSafeCall(cudaMemcpy(output, deviceOutput, length * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}

void WaveletTransformCudaImpl::reverse2D(const float *input, float *output, int width, int height, int levels, Wavelet *wavelet) {
    float * deviceInput;
    float * deviceOutput;
    int length = width * height;

    // Allocate device memory and copy input data
    cutilSafeCall(cudaMalloc((void**)&deviceInput, length * sizeof(float)));
    cutilSafeCall(cudaMalloc((void**)&deviceOutput, length * sizeof(float)));
    cutilSafeCall(cudaMemcpy(deviceOutput, input, length * sizeof(float), cudaMemcpyHostToDevice));

    // Perform reverse transform on device
    reverseDeviceMemory2D(deviceInput, deviceOutput, width, height, levels, wavelet);

    // Copy output data and free device memory
    cutilSafeCall(cudaMemcpy(output, deviceOutput, length * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}

void WaveletTransformCudaImpl::forward3D(const float *input, float *output, int width, int height, int frames, int levels, Wavelet *wavelet) {
    float * deviceInput;
    float * deviceOutput;
    int length = width * height * frames;

    // Allocate device memory and copy input data
    cutilSafeCall(cudaMalloc((void**)&deviceInput, length * sizeof(float)));
    cutilSafeCall(cudaMalloc((void**)&deviceOutput, length * sizeof(float)));
    cutilSafeCall(cudaMemcpy(deviceInput, input, length * sizeof(float), cudaMemcpyHostToDevice));

    // Perform forward transform on device
    forwardDeviceMemory3D(deviceInput, deviceOutput, width, height, frames, levels, wavelet);

    // Copy output data and free device memory
    cutilSafeCall(cudaMemcpy(output, deviceOutput, length * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}

void WaveletTransformCudaImpl::reverse3D(const float *input, float *output, int width, int height, int frames, int levels, Wavelet *wavelet) {
    float * deviceInput;
    float * deviceOutput;
    int length = width * height * frames;

    // Allocate device memory and copy input data
    cutilSafeCall(cudaMalloc((void**)&deviceInput, length * sizeof(float)));
    cutilSafeCall(cudaMalloc((void**)&deviceOutput, length * sizeof(float)));
    cutilSafeCall(cudaMemcpy(deviceInput, input, length * sizeof(float), cudaMemcpyHostToDevice));

    // Perform reverse transform on device
    reverseDeviceMemory3D(deviceInput, deviceOutput, width, height, frames, levels, wavelet);

    // Copy output data and free device memory
    cutilSafeCall(cudaMemcpy(output, deviceOutput, length * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}

void WaveletTransformCudaImpl::forwardDeviceMemory1D(float *deviceInput,
                                                     float *deviceOutput,
                                                     int length,
                                                     int levels,
                                                     Wavelet * wavelet) {
    int analysisLowLength       = wavelet->getAnalysisLowFilter()->getLength();
    int analysisHighLength      = wavelet->getAnalysisHighFilter()->getLength();
    int analysisLowFirstIndex   = wavelet->getAnalysisLowFilter()->getFirstIndex();
    int analysisHighFirstIndex  = wavelet->getAnalysisHighFilter()->getFirstIndex();
    int padding = getFilterPadding(analysisLowLength, analysisHighLength, analysisLowFirstIndex, analysisHighFirstIndex);
    int computeThreadCount = 2 * (THREADS_PER_BLOCK - padding);

    // Copy filter coefficients info constant memory buffer
    cutilSafeCall(cudaMemcpyToSymbol(lowPassFilter, wavelet->getAnalysisLowFilter()->getValues(), analysisLowLength * sizeof(float), 0));
    cutilSafeCall(cudaMemcpyToSymbol(highPassFilter, wavelet->getAnalysisHighFilter()->getValues(), analysisHighLength * sizeof(float), 0));

    int currLength = length;
    while(levels-- > 0) {
        int blockNum = getBlockNum(currLength, computeThreadCount);

        // Calcualte dwt on device
        forwardTransform1D<<<blockNum, THREADS_PER_BLOCK>>>(deviceInput, deviceOutput, currLength,
                                                            analysisLowLength, analysisHighLength,
                                                            analysisLowFirstIndex, analysisHighFirstIndex);
        cutilCheckMsg("kernel launch failure");

        currLength = (currLength + 1) / 2;

        cutilSafeCall(cudaMemcpy(deviceInput, deviceOutput, currLength * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

void WaveletTransformCudaImpl::reverseDeviceMemory1D(float *deviceInput,
                                                     float *deviceOutput,
                                                     int length,
                                                     int levels,
                                                     Wavelet * wavelet) {
    int synthesisLowLength      = wavelet->getSynthesisLowFilter()->getLength();
    int synthesisHighLength     = wavelet->getSynthesisHighFilter()->getLength();
    int synthesisLowFirstIndex  = wavelet->getSynthesisLowFilter()->getFirstIndex();
    int synthesisHighFirstIndex = wavelet->getSynthesisHighFilter()->getFirstIndex();
    int padding = getFilterPadding(synthesisLowLength, synthesisHighLength, synthesisLowFirstIndex, synthesisHighFirstIndex);
    int computeThreadCount = THREADS_PER_BLOCK - 2 * padding;

    // Copy filter coefficients info constant memory buffer
    cutilSafeCall(cudaMemcpyToSymbol(lowPassFilter, wavelet->getSynthesisLowFilter()->getValues(), synthesisLowLength * sizeof(float), 0));
    cutilSafeCall(cudaMemcpyToSymbol(highPassFilter, wavelet->getSynthesisHighFilter()->getValues(), synthesisHighLength * sizeof(float), 0));

    // Calculate signal length on specified decomposition level
    int currLength[levels];
    currLength[0] = length;
    for(int i = 1; i < levels; ++i) {
        currLength[i] = (currLength[i - 1] + 1) / 2;
    }

    while(levels-- > 0) {
        int blockNum = getBlockNum(currLength[levels], computeThreadCount);

        reverseTransform1D<<<blockNum, THREADS_PER_BLOCK>>>(deviceInput, deviceOutput, currLength[levels],
                                                            synthesisLowLength, synthesisHighLength,
                                                            synthesisLowFirstIndex, synthesisHighFirstIndex);
        cutilCheckMsg("kernel launch failure");

        cutilSafeCall(cudaMemcpy(deviceInput, deviceOutput, currLength[levels] * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

void WaveletTransformCudaImpl::forwardDeviceMemory2D(float *deviceInput,
                                                     float *deviceOutput,
                                                     int width,
                                                     int height,
                                                     int levels,
                                                     Wavelet *wavelet) {
    int analysisLowLength       = wavelet->getAnalysisLowFilter()->getLength();
    int analysisHighLength      = wavelet->getAnalysisHighFilter()->getLength();
    int analysisLowFirstIndex   = wavelet->getAnalysisLowFilter()->getFirstIndex();
    int analysisHighFirstIndex  = wavelet->getAnalysisHighFilter()->getFirstIndex();
    int padding = getFilterPadding(analysisLowLength, analysisHighLength, analysisLowFirstIndex, analysisHighFirstIndex);
    int computeThreadCount = 2 * (THREADS_PER_BLOCK - padding);

    // Copy filter coefficients info constant memory buffer
    cutilSafeCall(cudaMemcpyToSymbol(lowPassFilter, wavelet->getAnalysisLowFilter()->getValues(), analysisLowLength * sizeof(float), 0));
    cutilSafeCall(cudaMemcpyToSymbol(highPassFilter, wavelet->getAnalysisHighFilter()->getValues(), analysisHighLength * sizeof(float), 0));

    dim3 numBlocks, threadsPerBlock;
    int currWidth = width;
    int currHeight = height;
    while(levels-- > 0) {
        if((currWidth < 3 || currHeight < 3) && wavelet->isSymmetric()) {
            fail("Low pass subband to small, reduce levels.");
        }

        numBlocks.x = getBlockNum(currWidth, computeThreadCount);
        numBlocks.y = currHeight;
        threadsPerBlock.x = THREADS_PER_BLOCK;
        threadsPerBlock.y = 1;

        // Calcualte dwt on device for each row
        forwardTransform2DRow<<<numBlocks, threadsPerBlock>>>(deviceOutput, deviceInput, width, currWidth,
                                                              currHeight, analysisLowLength, analysisHighLength,
                                                              analysisLowFirstIndex, analysisHighFirstIndex);
        cutilCheckMsg("kernel launch failure");

        numBlocks.x = currWidth;
        numBlocks.y = getBlockNum(currHeight, computeThreadCount);
        threadsPerBlock.x = 1;
        threadsPerBlock.y = THREADS_PER_BLOCK;

        // Calcualte dwt on device for each column
        forwardTransform2DColumn<<<numBlocks, threadsPerBlock>>>(deviceInput, deviceOutput, width, currWidth,
                                                                 currHeight, analysisLowLength, analysisHighLength,
                                                                 analysisLowFirstIndex, analysisHighFirstIndex);
        cutilCheckMsg("kernel launch failure");

        currWidth = (currWidth+1) / 2;
        currHeight = (currHeight+1) / 2;
    }
}

void WaveletTransformCudaImpl::reverseDeviceMemory2D(float *deviceInput,
                                                     float *deviceOutput,
                                                     int width,
                                                     int height,
                                                     int levels,
                                                     Wavelet *wavelet) {
    int synthesisLowLength      = wavelet->getSynthesisLowFilter()->getLength();
    int synthesisHighLength     = wavelet->getSynthesisHighFilter()->getLength();
    int synthesisLowFirstIndex  = wavelet->getSynthesisLowFilter()->getFirstIndex();
    int synthesisHighFirstIndex = wavelet->getSynthesisHighFilter()->getFirstIndex();
    int padding = getFilterPadding(synthesisLowLength, synthesisHighLength, synthesisLowFirstIndex, synthesisHighFirstIndex);
    int computeThreadCount = THREADS_PER_BLOCK - 2 * padding;

    // Copy filter coefficients info constant memory buffer
    cutilSafeCall(cudaMemcpyToSymbol(lowPassFilter, wavelet->getSynthesisLowFilter()->getValues(), synthesisLowLength * sizeof(float), 0));
    cutilSafeCall(cudaMemcpyToSymbol(highPassFilter, wavelet->getSynthesisHighFilter()->getValues(), synthesisHighLength * sizeof(float), 0));

    int currWidth[levels];
    int currHeight[levels];

    currWidth[0] = width;
    for(int i = 1; i < levels; ++i) {
        currWidth[i] = (currWidth[i - 1] + 1) / 2;
    }

    currHeight[0] = height;
    for(int i = 1; i < levels; ++i) {
        currHeight[i] = (currHeight[i - 1] + 1) / 2;
    }

    dim3 numBlocks, threadsPerBlock;
    while(levels-- > 0) {
        numBlocks.x = getBlockNum(currWidth[levels], computeThreadCount);
        numBlocks.y = currHeight[levels];
        threadsPerBlock.x = THREADS_PER_BLOCK;
        threadsPerBlock.y = 1;

        // Calcualte idwt on device for each row
        reverseTransform2DRow<<<numBlocks, threadsPerBlock>>>(deviceOutput, deviceInput, width, currWidth[levels],
                                                              currHeight[levels], synthesisLowLength, synthesisHighLength,
                                                              synthesisLowFirstIndex, synthesisHighFirstIndex);
        cutilCheckMsg("kernel launch failure");

        numBlocks.x = currWidth[levels];
        numBlocks.y = getBlockNum(currHeight[levels], computeThreadCount);
        threadsPerBlock.x = 1;
        threadsPerBlock.y = THREADS_PER_BLOCK;

        // Calcualte idwt on device for each column
        reverseTransform2DColumn<<<numBlocks, threadsPerBlock>>>(deviceInput, deviceOutput, width, currWidth[levels],
                                                                 currHeight[levels], synthesisLowLength, synthesisHighLength,
                                                                 synthesisLowFirstIndex, synthesisHighFirstIndex);
        cutilCheckMsg("kernel launch failure");
    }
}

void WaveletTransformCudaImpl::forwardDeviceMemory3D(float *deviceInput,
                                                     float *deviceOutput,
                                                     int width,
                                                     int height,
                                                     int frames,
                                                     int levels,
                                                     Wavelet *wavelet) {
    int analysisLowLength       = wavelet->getAnalysisLowFilter()->getLength();
    int analysisHighLength      = wavelet->getAnalysisHighFilter()->getLength();
    int analysisLowFirstIndex   = wavelet->getAnalysisLowFilter()->getFirstIndex();
    int analysisHighFirstIndex  = wavelet->getAnalysisHighFilter()->getFirstIndex();
    int padding = getFilterPadding(analysisLowLength, analysisHighLength, analysisLowFirstIndex, analysisHighFirstIndex);
    int computeThreadCount = 2 * (THREADS_PER_BLOCK - padding);

    // Copy filter coefficients info constant memory buffer
    cutilSafeCall(cudaMemcpyToSymbol(lowPassFilter, wavelet->getAnalysisLowFilter()->getValues(), analysisLowLength * sizeof(float), 0));
    cutilSafeCall(cudaMemcpyToSymbol(highPassFilter, wavelet->getAnalysisHighFilter()->getValues(), analysisHighLength * sizeof(float), 0));

    dim3 numBlocks, threadsPerBlock;
    int currWidth = width;
    int currHeight = height;
    int currDepth = frames;
    while(levels-- > 0) {
        if((currWidth < 3 || currHeight < 3 || currDepth < 3) && wavelet->isSymmetric()) {
            fail("Low pass subband to small, reduce levels.");
        }

        numBlocks.x = getBlockNum(currWidth, computeThreadCount) * currHeight;
        numBlocks.y = currDepth;
        threadsPerBlock.x = THREADS_PER_BLOCK;

        // Calcualte dwt on device for each row
        forwardTransform3DRow<<<numBlocks, threadsPerBlock>>>(deviceInput, deviceOutput, width, height, currWidth,
                                                              currHeight, currDepth, analysisLowLength,
                                                              analysisHighLength, analysisLowFirstIndex,
                                                              analysisHighFirstIndex);
        cutilCheckMsg("kernel launch failure");

        numBlocks.x = getBlockNum(currHeight, computeThreadCount) * currWidth;
        numBlocks.y = currDepth;

        // Calcualte dwt on device for each column
        forwardTransform3DColumn<<<numBlocks, threadsPerBlock>>>(deviceOutput, deviceInput, width, height, currWidth,
                                                                 currHeight, currDepth, analysisLowLength,
                                                                 analysisHighLength, analysisLowFirstIndex,
                                                                 analysisHighFirstIndex);
        cutilCheckMsg("kernel launch failure");

        numBlocks.x = getBlockNum(currDepth, computeThreadCount) * currWidth;
        numBlocks.y = currHeight;

        // Calcualte dwt on device for each frame
        forwardTransform3DFrame<<<numBlocks, threadsPerBlock>>>(deviceInput, deviceOutput, width, height, currWidth,
                                                                currHeight, currDepth, analysisLowLength,
                                                                analysisHighLength, analysisLowFirstIndex,
                                                                analysisHighFirstIndex);
        cutilCheckMsg("kernel launch failure");

        cutilSafeCall(cudaMemcpy(deviceInput, deviceOutput, width * height * frames * sizeof(float),
                                 cudaMemcpyDeviceToDevice));

        currWidth   = (currWidth+1) / 2;
        currHeight  = (currHeight+1) / 2;
        currDepth   = (currDepth+1) / 2;
    }
}

void WaveletTransformCudaImpl::reverseDeviceMemory3D(float *deviceInput,
                                                     float *deviceOutput,
                                                     int width,
                                                     int height,
                                                     int frames,
                                                     int levels,
                                                     Wavelet *wavelet) {
    int synthesisLowLength      = wavelet->getSynthesisLowFilter()->getLength();
    int synthesisHighLength     = wavelet->getSynthesisHighFilter()->getLength();
    int synthesisLowFirstIndex  = wavelet->getSynthesisLowFilter()->getFirstIndex();
    int synthesisHighFirstIndex = wavelet->getSynthesisHighFilter()->getFirstIndex();
    int padding = getFilterPadding(synthesisLowLength, synthesisHighLength, synthesisLowFirstIndex, synthesisHighFirstIndex);
    int computeThreadCount = THREADS_PER_BLOCK - 2 * padding;

    // Copy filter coefficients info constant memory buffer
    cutilSafeCall(cudaMemcpyToSymbol(lowPassFilter, wavelet->getSynthesisLowFilter()->getValues(), synthesisLowLength * sizeof(float), 0));
    cutilSafeCall(cudaMemcpyToSymbol(highPassFilter, wavelet->getSynthesisHighFilter()->getValues(), synthesisHighLength * sizeof(float), 0));

    int currWidth[levels];
    int currHeight[levels];
    int currDepth[levels];

    currWidth[0] = width;
    for(int i = 1; i < levels; ++i) {
        currWidth[i] = (currWidth[i - 1] + 1) / 2;
    }

    currHeight[0] = height;
    for(int i = 1; i < levels; ++i) {
        currHeight[i] = (currHeight[i - 1] + 1) / 2;
    }

    currDepth[0] = frames;
    for(int i = 1; i < levels; ++i) {
        currDepth[i] = (currDepth[i - 1] + 1) / 2;
    }

    cutilSafeCall(cudaMemcpy(deviceOutput, deviceInput, width * height * frames * sizeof(float),
                             cudaMemcpyDeviceToDevice));

    dim3 numBlocks, threadsPerBlock;
    while(levels-- > 0) {
        numBlocks.x = getBlockNum(currWidth[levels], computeThreadCount) * currHeight[levels];
        numBlocks.y = currDepth[levels];
        threadsPerBlock.x = THREADS_PER_BLOCK;

        // Calcualte idwt on device for each row
        reverseTransform3DRow<<<numBlocks, threadsPerBlock>>>(deviceInput, deviceOutput, width, height, currWidth[levels],
                                                              currHeight[levels], currDepth[levels], synthesisLowLength,
                                                              synthesisHighLength, synthesisLowFirstIndex,
                                                              synthesisHighFirstIndex);
        cutilCheckMsg("kernel launch failure");

        numBlocks.x = getBlockNum(currHeight[levels], computeThreadCount) * currWidth[levels];
        numBlocks.y = currDepth[levels];

        // Calcualte idwt on device for each column
        reverseTransform3DColumn<<<numBlocks, threadsPerBlock>>>(deviceOutput, deviceInput, width, height, currWidth[levels],
                                                                 currHeight[levels], currDepth[levels], synthesisLowLength,
                                                                 synthesisHighLength, synthesisLowFirstIndex,
                                                                 synthesisHighFirstIndex);
        cutilCheckMsg("kernel launch failure");

        numBlocks.x = getBlockNum(currDepth[levels], computeThreadCount) * currWidth[levels];
        numBlocks.y = currHeight[levels];

        // Calcualte idwt on device for each frame
        reverseTransform3DFrame<<<numBlocks, threadsPerBlock>>>(deviceInput, deviceOutput, width, height, currWidth[levels],
                                                                currHeight[levels], currDepth[levels], synthesisLowLength,
                                                                synthesisHighLength, synthesisLowFirstIndex,
                                                                synthesisHighFirstIndex);
        cutilCheckMsg("kernel launch failure");

        cutilSafeCall(cudaMemcpy(deviceInput, deviceOutput, width * height * frames * sizeof(float),
                                 cudaMemcpyDeviceToDevice));
    }
}
