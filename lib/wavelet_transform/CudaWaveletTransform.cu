/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include <cutil_inline.h>

const int THREADS_PER_BLOCK = 256;  // Number of threads per block
const int FILTER_BUFF_LENGTH = 64;  // Length of constant memory buffer used for filters coefficients

const int FIXED_ROW     = 0;
const int FIXED_COLUMN  = 1;
const int FIXED_FRAME   = 2;

__constant__ float lowPassFilter[FILTER_BUFF_LENGTH];   // Constant memory buffer used for high pass filter coefficients
__constant__ float highPassFilter[FILTER_BUFF_LENGTH];  // Constant memory buffer used for low pass filter coefficients

__device__ inline int getMappedIndex(int index,
                                     int inputLength,
                                     bool ignoreOddIndex,
                                     bool ignoreEvenIndex) {
    int newIndex = -1;

    // check if index is in range [0, length)
    if(index >= 0 && index < inputLength) {
        newIndex = index;
    }
    else {
        if(index < 0) {
            while(index < -inputLength) {
                index += inputLength;
            }
            newIndex = inputLength + index;
        }
        else if(index >= inputLength) {
            newIndex = index;
            while(newIndex >= inputLength) {
                newIndex = newIndex - inputLength;
            }
        }
    }

    if(ignoreOddIndex) {
        if(abs(newIndex) % 2 == 1) {
            return -1;
        }
        return newIndex;
    }

    if(ignoreEvenIndex) {
        if(abs(newIndex) % 2 == 0) {
            return -1;
        }
        return newIndex;
    }

    return newIndex;
}

__device__ inline float getInputValue1D(const float *input,
                                        int index,
                                        int inputLength,
                                        int offset = 0,
                                        bool ignoreOddIndex = false,
                                        bool ignoreEvenIndex = false) {
    int newIndex = getMappedIndex(index, inputLength, ignoreOddIndex, ignoreEvenIndex);

    if(newIndex < 0) {
        return 0.0f;
    }

    if(ignoreOddIndex || ignoreEvenIndex) {
        return input[newIndex/2 + offset];
    }

    return input[newIndex];
}

__device__ inline float getInputValue2D(const float *input,
                                        int x,
                                        int y,
                                        bool fixedRow,
                                        int width,
                                        int currWidth,
                                        int currHeight,
                                        int offset = 0,
                                        bool ignoreOddIndex = false,
                                        bool ignoreEvenIndex = false) {
    int newIndex = getMappedIndex(fixedRow ? x : y, fixedRow ? currWidth : currHeight,
                                  ignoreOddIndex, ignoreEvenIndex);

    if(newIndex < 0) {
        return 0.0f;
    }

    if(ignoreOddIndex || ignoreEvenIndex) {
        if(fixedRow) {
            return input[y * width + newIndex/2 + offset];
        }
        return input[(newIndex/2 + offset) * width + x];
    }

    return fixedRow ? input[y * width + newIndex] : input[newIndex * width + x];
}

__device__ inline float getInputValue3D(const float *input,
                                        int x,
                                        int y,
                                        int z,
                                        int fixedDim,
                                        int width,
                                        int height,
                                        int currWidth,
                                        int currHeight,
                                        int currFrames,
                                        int offset = 0,
                                        bool ignoreOddIndex = false,
                                        bool ignoreEvenIndex = false) {
    int newIndex =  -1;
    if(fixedDim == FIXED_ROW) {
        newIndex = getMappedIndex(x, currWidth, ignoreOddIndex, ignoreEvenIndex);
    }
    else if(fixedDim == FIXED_COLUMN) {
        newIndex = getMappedIndex(y, currHeight, ignoreOddIndex, ignoreEvenIndex);
    }
    else {
        newIndex = getMappedIndex(z, currFrames, ignoreOddIndex, ignoreEvenIndex);
    }

    if(newIndex < 0) {
        return 0.0f;
    }

    if(ignoreOddIndex || ignoreEvenIndex) {
        if(fixedDim == FIXED_ROW) {
            return input[z * width * height + y * width + newIndex/2 + offset];
        } else if(fixedDim == FIXED_COLUMN) {
            return input[z * width * height + (newIndex/2 + offset) * width + x];
        } else {
            return input[(newIndex/2 + offset) * width * height + y * width + x];
        }
    }

    if(fixedDim == FIXED_ROW) {
        return input[z * width * height + y * width + newIndex];
    } else if(fixedDim == FIXED_COLUMN) {
        return input[z * width * height + newIndex * width + x];
    }
    return input[newIndex * width * height + y * width + x];
}

__device__ inline float forwardStepLow(float *input,
                                       int inputOffset,
                                       int i,
                                       int analysisLowLength,
                                       int analysisLowFirstIndex) {
    float value   = 0.0f;

    // Convolve with low pass analysis filter - aproximation coefficient
    for(int j = 0; j < analysisLowLength; ++j) {
        int k = (2 * i) + j - analysisLowFirstIndex;
        float inputValue = input[k - inputOffset];
        if(inputValue == 0) {
            continue;
        }

        value += inputValue * lowPassFilter[j];
    }
    return value;
}

__device__ inline float forwardStepHigh(float *input,
                                        int inputOffset,
                                        int i,
                                        int analysisHighLength,
                                        int analysisHighFirstIndex) {
    float value   = 0.0f;

    // Convolve with high pass analysis filter - detail coefficient
    for(int j = 0; j < analysisHighLength; ++j) {
        int k = (2 * i) + j - analysisHighFirstIndex;
        float inputValue = input[k - inputOffset];
        if(inputValue == 0) {
            continue;
        }

        value += inputValue * highPassFilter[j];
    }
    return value;
}

__device__ inline float reverseStep(float *lowInput,
                                    float *highInput,
                                    int inputOffset,
                                    int i,
                                    int synthesisLowLength,
                                    int synthesisHighLength,
                                    int synthesisLowFirstIndex,
                                    int synthesisHighFirstIndex) {
    float value = 0.0f;

    // Convolve with low pass synthesis filter
    for(int j = 0; j < synthesisLowLength; ++j) {
        int k = i - j + synthesisLowFirstIndex;
        float inputValue = lowInput[k - inputOffset];
        if(inputValue == 0) {
            continue;
        }

        value += inputValue * lowPassFilter[j];
    }

    // Convolve with high pass synthesis filter
    for(int j = 0; j < synthesisHighLength; ++j) {
        int k = i - j + synthesisHighFirstIndex;
        float inputValue = highInput[k - inputOffset];
        if(inputValue == 0) {
            continue;
        }

        value += inputValue * highPassFilter[j];
    }

    return value;
}

__device__ __host__ inline int getFilterPadding(int filterLowLength,
                                                int filterHighLength,
                                                int filterLowFirstIndex,
                                                int filterHighFirstIndex) {
    int padding = filterLowLength > filterHighLength ? filterLowLength : filterHighLength;
    padding -= filterLowFirstIndex < filterHighFirstIndex ? filterLowFirstIndex : filterHighFirstIndex;
    return padding;
}

__device__ __host__ inline int getBlockNum(int dataLenght, int blockSize) {
    if(dataLenght < blockSize) {
        return 1;
    }
    int blockNum = dataLenght / blockSize;
    blockNum += dataLenght % blockSize == 0 ? 0 : 1;
    return blockNum;
}

__global__ void forwardTransform1D(float *input,
                                   float *output,
                                   int length,
                                   int analysisLowLength,
                                   int analysisHighLength,
                                   int analysisLowFirstIndex,
                                   int analysisHighFirstIndex) {
    // Calculate filter padding
    int padding = getFilterPadding(analysisLowLength, analysisHighLength, analysisLowFirstIndex, analysisHighFirstIndex);
    int computeThreadGroupSize      = THREADS_PER_BLOCK - padding;              // number of threads used to calculate dwt
    int computeThreadGroupOffset    = 2 * blockIdx.x * computeThreadGroupSize;  // index of first compute thread
    int threadLoadGroupOffset       = computeThreadGroupOffset - padding;       // index of first load thread
    int i = computeThreadGroupOffset / 2 + threadIdx.x - padding;               // current thread compute element index

    // Load input data to shared memory; Each thread loads two elements
    __shared__ float sharedInput[2 * THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];

    sharedInput[2 * threadIdx.x]     = getInputValue1D(input, threadLoadGroupOffset + 2 * threadIdx.x, length);
    sharedInput[2 * threadIdx.x + 1] = getInputValue1D(input, threadLoadGroupOffset + 2 * threadIdx.x + 1, length);

    __syncthreads();

    // Threads with id lower than padding are used only for loading data into shared memory
    if(threadIdx.x < padding) {
        return;
    }

    int lowLength   = (length + 1) / 2;   // Low subband length
    int inputOffset = computeThreadGroupOffset - padding;

    // Check if outcome index is lower than low subband length
    if(i >= lowLength) {
        return;
    }

    output[i] = forwardStepLow(sharedInput, inputOffset, i, analysisLowLength, analysisLowFirstIndex);

    // Check if outcome index is lower than low subband length
    if(i + lowLength >= length) {
        return;
    }

    output[i + lowLength] = forwardStepHigh(sharedInput, inputOffset, i, analysisHighLength, analysisHighFirstIndex);
}

__global__ void reverseTransform1D(float *input,
                                   float *output,
                                   int length,
                                   int synthesisLowLength,
                                   int synthesisHighLength,
                                   int synthesisLowFirstIndex,
                                   int synthesisHighFirstIndex) {
    // Calculate filter padding
    int padding = getFilterPadding(synthesisLowLength, synthesisHighLength, synthesisLowFirstIndex, synthesisHighFirstIndex);
    int threadComputeGroupOffset    = blockIdx.x * (THREADS_PER_BLOCK - 2 * padding);   // index of first compute thread
    int threadLoadGroupOffset       = threadComputeGroupOffset - padding;               // index of first load thread
    int loadThreadId                = threadLoadGroupOffset + threadIdx.x;              // current thread load element index
    int i = threadComputeGroupOffset + threadIdx.x - padding;                           // current thread compute element index

    bool analysisIgnoreEven = synthesisHighLength % 2 == 0; // Check wether ignore even or odd input values
    int lowLength           = (length + 1) / 2;             // Low subband length
    int highLength          = length - lowLength;           // High subband length
    int inputOffset         = threadComputeGroupOffset - padding;

    // Load input data to shared memory; Each thread one element for low pass and one element for high pass
    __shared__ float lowInput [THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];
    __shared__ float highInput[THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];

    lowInput[threadIdx.x]  = getInputValue1D(input, loadThreadId, 2 * lowLength, 0, true);
    highInput[threadIdx.x] = getInputValue1D(input, loadThreadId, 2 * highLength, lowLength,
                                             analysisIgnoreEven, !analysisIgnoreEven);

    __syncthreads();

    // Check if outcome index is lower than low subband length and if thread is compute thread
    // (not used only to load data into shared memory)
    if(i >= length || threadIdx.x < padding || threadIdx.x >= THREADS_PER_BLOCK - padding) {
        return;
    }

    output[i] = reverseStep(lowInput, highInput, inputOffset, i, synthesisLowLength, synthesisHighLength,
                            synthesisLowFirstIndex, synthesisHighFirstIndex);
}

__global__ void forwardTransform2DRow(float *input,
                                      float *output,
                                      int width,
                                      int currWidth,
                                      int currHeight,
                                      int analysisLowLength,
                                      int analysisHighLength,
                                      int analysisLowFirstIndex,
                                      int analysisHighFirstIndex) {
    // Calculate filter padding
    int padding = getFilterPadding(analysisLowLength, analysisHighLength, analysisLowFirstIndex, analysisHighFirstIndex);
    int computeThreadGroupSize      = THREADS_PER_BLOCK - padding;              // number of threads used to calculate dwt
    int computeThreadGroupOffset    = 2 * blockIdx.x * computeThreadGroupSize;  // index of first compute thread
    int threadLoadGroupOffset       = computeThreadGroupOffset - padding;       // index of first load thread
    int x = computeThreadGroupOffset / 2 + threadIdx.x - padding;               // current thread compute element x index
    int y = blockIdx.y;
    int threadLoadIndex = threadLoadGroupOffset + 2 * threadIdx.x;

    // Load input data to shared memory; Each thread loads two elements
    __shared__ float sharedInput[2 * THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];

    sharedInput[2 * threadIdx.x]     = getInputValue2D(input, threadLoadIndex, y, true, width, currWidth, currHeight);
    sharedInput[2 * threadIdx.x + 1] = getInputValue2D(input, threadLoadIndex + 1, y, true, width, currWidth, currHeight);

    __syncthreads();

    // Threads with id lower than padding are used only for loading data into shared memory
    if(threadIdx.x < padding) {
        return;
    }

    int lowLength   = (currWidth + 1) / 2;   // Low subband length
    int inputOffset = computeThreadGroupOffset - padding;

    // Check if outcome index is lower than low subband length
    if(x >= lowLength) {
        return;
    }

    output[y * width + x] = forwardStepLow(sharedInput, inputOffset, x, analysisLowLength, analysisLowFirstIndex);

    // Check if outcome index is lower than low subband length
    if(x + lowLength >= currWidth) {
        return;
    }

    output[y * width + x + lowLength] = forwardStepHigh(sharedInput, inputOffset, x, analysisHighLength, analysisHighFirstIndex);
}

__global__ void forwardTransform2DColumn(float *input,
                                         float *output,
                                         int width,
                                         int currWidth,
                                         int currHeight,
                                         int analysisLowLength,
                                         int analysisHighLength,
                                         int analysisLowFirstIndex,
                                         int analysisHighFirstIndex) {
    // Calculate filter padding
    int padding = getFilterPadding(analysisLowLength, analysisHighLength, analysisLowFirstIndex, analysisHighFirstIndex);
    int computeThreadGroupSize      = THREADS_PER_BLOCK - padding;              // number of threads used to calculate dwt
    int computeThreadGroupOffset    = 2 * blockIdx.y * computeThreadGroupSize;  // index of first compute thread
    int threadLoadGroupOffset       = computeThreadGroupOffset - padding;       // index of first load thread
    int y = computeThreadGroupOffset / 2 + threadIdx.y - padding;               // current thread compute element y index
    int x = blockIdx.x;
    int threadLoadIndex = threadLoadGroupOffset + 2 * threadIdx.y;

    // Load input data to shared memory; Each thread loads two elements
    __shared__ float sharedInput[2 * THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];

    sharedInput[2 * threadIdx.y]     = getInputValue2D(input, x, threadLoadIndex, false, width, currWidth, currHeight);
    sharedInput[2 * threadIdx.y + 1] = getInputValue2D(input, x, threadLoadIndex + 1, false, width, currWidth, currHeight);

    __syncthreads();

    // Threads with id lower than padding are used only for loading data into shared memory
    if(threadIdx.y < padding) {
        return;
    }

    int lowLength   = (currHeight + 1) / 2;   // Low subband length
    int inputOffset = computeThreadGroupOffset - padding;

    // Check if outcome index is lower than low subband length
    if(y >= lowLength) {
        return;
    }

    output[y * width + x] = forwardStepLow(sharedInput, inputOffset, y, analysisLowLength, analysisLowFirstIndex);

    // Check if outcome index is lower than low subband length
    if(y + lowLength >= currHeight) {
        return;
    }

    output[(y + lowLength) * width + x] = forwardStepHigh(sharedInput, inputOffset, y, analysisHighLength, analysisHighFirstIndex);
}

__global__ void reverseTransform2DRow(float *input,
                                      float *output,
                                      int width,
                                      int currWidth,
                                      int currHeight,
                                      int synthesisLowLength,
                                      int synthesisHighLength,
                                      int synthesisLowFirstIndex,
                                      int synthesisHighFirstIndex) {
    // Calculate filter padding
    int padding = getFilterPadding(synthesisLowLength, synthesisHighLength, synthesisLowFirstIndex, synthesisHighFirstIndex);
    int threadComputeGroupOffset    = blockIdx.x * (THREADS_PER_BLOCK - 2 * padding);   // index of first compute thread
    int threadLoadGroupOffset       = threadComputeGroupOffset - padding;               // index of first load thread
    int loadThreadId                = threadLoadGroupOffset + threadIdx.x;              // current thread load element index
    int i = threadComputeGroupOffset + threadIdx.x - padding;                           // current thread compute element index
    int y = blockIdx.y;

    bool analysisIgnoreEven = synthesisHighLength % 2 == 0; // Check wether ignore even or odd input values
    int lowLength           = (currWidth + 1) / 2;          // Low subband length
    int highLength          = currWidth - lowLength;        // High subband length
    int inputOffset         = threadComputeGroupOffset - padding;

    // Load input data to shared memory; Each thread one element for low pass and one element for high pass
    __shared__ float lowInput [THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];
    __shared__ float highInput[THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];

    lowInput[threadIdx.x]  = getInputValue2D(input, loadThreadId, y, true, width, 2 * lowLength, currHeight, 0, true);
    highInput[threadIdx.x] = getInputValue2D(input, loadThreadId, y, true, width, 2 * highLength, currHeight, lowLength,
                                             analysisIgnoreEven, !analysisIgnoreEven);

    __syncthreads();

    // Check if outcome index is lower than low subband length and if thread is compute thread
    // (not used only to load data into shared memory)
    if(i >= currWidth || threadIdx.x < padding || threadIdx.x >= THREADS_PER_BLOCK - padding) {
        return;
    }

    output[y * width + i] = reverseStep(lowInput, highInput, inputOffset, i, synthesisLowLength, synthesisHighLength,
                                        synthesisLowFirstIndex, synthesisHighFirstIndex);
}

__global__ void reverseTransform2DColumn(float *input,
                                         float *output,
                                         int width,
                                         int currWidth,
                                         int currHeight,
                                         int synthesisLowLength,
                                         int synthesisHighLength,
                                         int synthesisLowFirstIndex,
                                         int synthesisHighFirstIndex) {
    // Calculate filter padding
    int padding = getFilterPadding(synthesisLowLength, synthesisHighLength, synthesisLowFirstIndex, synthesisHighFirstIndex);
    int threadComputeGroupOffset    = blockIdx.y* (THREADS_PER_BLOCK - 2 * padding);   // index of first compute thread
    int threadLoadGroupOffset       = threadComputeGroupOffset - padding;               // index of first load thread
    int loadThreadId                = threadLoadGroupOffset + threadIdx.y;              // current thread load element index
    int y = threadComputeGroupOffset + threadIdx.y - padding;                           // current thread compute element index
    int x = blockIdx.x;

    bool analysisIgnoreEven = synthesisHighLength % 2 == 0; // Check wether ignore even or odd input values
    int lowLength           = (currHeight + 1) / 2;          // Low subband length
    int highLength          = currHeight - lowLength;        // High subband length
    int inputOffset         = threadComputeGroupOffset - padding;

    // Load input data to shared memory; Each thread one element for low pass and one element for high pass
    __shared__ float lowInput [THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];
    __shared__ float highInput[THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];

    lowInput[threadIdx.y]  = getInputValue2D(input, x, loadThreadId, false, width, currWidth, 2 * lowLength, 0, true);
    highInput[threadIdx.y] = getInputValue2D(input, x, loadThreadId, false, width, currWidth, 2 * highLength, lowLength,
                                             analysisIgnoreEven, !analysisIgnoreEven);

    __syncthreads();

    // Check if outcome index is lower than low subband length and if thread is compute thread
    // (not used only to load data into shared memory)
    if(y >= currHeight || threadIdx.y < padding || threadIdx.y >= THREADS_PER_BLOCK - padding) {
        return;
    }

    output[y * width + x] = reverseStep(lowInput, highInput, inputOffset, y, synthesisLowLength, synthesisHighLength,
                                        synthesisLowFirstIndex, synthesisHighFirstIndex);
}

__global__ void forwardTransform3DRow(float *input,
                                      float *output,
                                      int width,
                                      int height,
                                      int currWidth,
                                      int currHeight,
                                      int currFrames,
                                      int analysisLowLength,
                                      int analysisHighLength,
                                      int analysisLowFirstIndex,
                                      int analysisHighFirstIndex) {
    // Calculate filter padding
    int padding = getFilterPadding(analysisLowLength, analysisHighLength, analysisLowFirstIndex, analysisHighFirstIndex);
    int computeThreadCount = 2 * (THREADS_PER_BLOCK - padding);
    int blockNumX = getBlockNum(currWidth, computeThreadCount);
    int computeThreadGroupSize      = THREADS_PER_BLOCK - padding;              // number of threads used to calculate dwt
    int computeThreadGroupOffset    = 2 * (blockIdx.x % blockNumX) * computeThreadGroupSize;  // index of first compute thread
    int threadLoadGroupOffset       = computeThreadGroupOffset - padding;       // index of first load thread
    int x = computeThreadGroupOffset / 2 + threadIdx.x - padding;               // current thread compute element x index
    int y = blockIdx.x / blockNumX;
    int z = blockIdx.y;
    int threadLoadIndex = threadLoadGroupOffset + 2 * threadIdx.x;

    // Load input data to shared memory; Each thread loads two elements
    __shared__ float sharedInput[2 * THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];

    sharedInput[2 * threadIdx.x]     = getInputValue3D(input, threadLoadIndex, y, z, FIXED_ROW, width, height,
                                                       currWidth, currHeight, currFrames);
    sharedInput[2 * threadIdx.x + 1] = getInputValue3D(input, threadLoadIndex + 1, y, z, FIXED_ROW, width, height,
                                                       currWidth, currHeight, currFrames);
    __syncthreads();

    // Threads with id lower than padding are used only for loading data into shared memory
    if(threadIdx.x < padding) {
        return;
    }

    int lowLength   = (currWidth + 1) / 2;   // Low subband length
    int inputOffset = computeThreadGroupOffset - padding;

    // Check if outcome index is lower than low subband length
    if(x >= lowLength) {
        return;
    }

    output[z * width * height + y * width + x] = forwardStepLow(sharedInput, inputOffset, x, analysisLowLength,
                                                                analysisLowFirstIndex);

    // Check if outcome index is lower than low subband length
    if(x + lowLength >= currWidth) {
        return;
    }

    output[z * width * height + y * width + x + lowLength] = forwardStepHigh(sharedInput, inputOffset, x,
                                                                             analysisHighLength, analysisHighFirstIndex);
}

__global__ void forwardTransform3DColumn(float *input,
                                         float *output,
                                         int width,
                                         int height,
                                         int currWidth,
                                         int currHeight,
                                         int currFrames,
                                         int analysisLowLength,
                                         int analysisHighLength,
                                         int analysisLowFirstIndex,
                                         int analysisHighFirstIndex) {
    // Calculate filter padding
    int padding = getFilterPadding(analysisLowLength, analysisHighLength, analysisLowFirstIndex, analysisHighFirstIndex);
    int computeThreadCount = 2 * (THREADS_PER_BLOCK - padding);
    int blockNumX = getBlockNum(currHeight, computeThreadCount);
    int computeThreadGroupSize      = THREADS_PER_BLOCK - padding;              // number of threads used to calculate dwt
    int computeThreadGroupOffset    = 2 * (blockIdx.x % blockNumX) * computeThreadGroupSize;  // index of first compute thread
    int threadLoadGroupOffset       = computeThreadGroupOffset - padding;       // index of first load thread
    int y = computeThreadGroupOffset / 2 + threadIdx.x - padding;               // current thread compute element x index
    int x = blockIdx.x / blockNumX;
    int z = blockIdx.y;
    int threadLoadIndex = threadLoadGroupOffset + 2 * threadIdx.x;

    // Load input data to shared memory; Each thread loads two elements
    __shared__ float sharedInput[2 * THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];

    sharedInput[2 * threadIdx.x]     = getInputValue3D(input, x, threadLoadIndex, z, FIXED_COLUMN, width, height,
                                                       currWidth, currHeight, currFrames);
    sharedInput[2 * threadIdx.x + 1] = getInputValue3D(input, x, threadLoadIndex + 1, z, FIXED_COLUMN, width, height,
                                                       currWidth, currHeight, currFrames);
    __syncthreads();

    // Threads with id lower than padding are used only for loading data into shared memory
    if(threadIdx.x < padding) {
        return;
    }

    int lowLength   = (currHeight + 1) / 2;   // Low subband length
    int inputOffset = computeThreadGroupOffset - padding;

    // Check if outcome index is lower than low subband length
    if(y >= lowLength) {
        return;
    }

    output[z * width * height + y * width + x] = forwardStepLow(sharedInput, inputOffset, y, analysisLowLength,
                                                                analysisLowFirstIndex);

    // Check if outcome index is lower than low subband length
    if(y + lowLength >= currHeight) {
        return;
    }

    output[z * width * height + (y + lowLength) * width + x] = forwardStepHigh(sharedInput, inputOffset, y,
                                                                               analysisHighLength, analysisHighFirstIndex);
}

__global__ void forwardTransform3DFrame(float *input,
                                        float *output,
                                        int width,
                                        int height,
                                        int currWidth,
                                        int currHeight,
                                        int currFrames,
                                        int analysisLowLength,
                                        int analysisHighLength,
                                        int analysisLowFirstIndex,
                                        int analysisHighFirstIndex) {
    // Calculate filter padding
    int padding = getFilterPadding(analysisLowLength, analysisHighLength, analysisLowFirstIndex, analysisHighFirstIndex);
    int computeThreadCount = 2 * (THREADS_PER_BLOCK - padding);
    int blockNumX = getBlockNum(currFrames, computeThreadCount);
    int computeThreadGroupSize      = THREADS_PER_BLOCK - padding;              // number of threads used to calculate dwt
    int computeThreadGroupOffset    = 2 * (blockIdx.x % blockNumX) * computeThreadGroupSize;  // index of first compute thread
    int threadLoadGroupOffset       = computeThreadGroupOffset - padding;       // index of first load thread
    int z = computeThreadGroupOffset / 2 + threadIdx.x - padding;               // current thread compute element x index
    int x = blockIdx.x / blockNumX;
    int y = blockIdx.y;
    int threadLoadIndex = threadLoadGroupOffset + 2 * threadIdx.x;

    // Load input data to shared memory; Each thread loads two elements
    __shared__ float sharedInput[2 * THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];

    sharedInput[2 * threadIdx.x]     = getInputValue3D(input, x, y, threadLoadIndex, FIXED_FRAME, width, height,
                                                       currWidth, currHeight, currFrames);
    sharedInput[2 * threadIdx.x + 1] = getInputValue3D(input, x, y, threadLoadIndex + 1, FIXED_FRAME, width, height,
                                                       currWidth, currHeight, currFrames);
    __syncthreads();

    // Threads with id lower than padding are used only for loading data into shared memory
    if(threadIdx.x < padding) {
        return;
    }

    int lowLength   = (currFrames + 1) / 2;   // Low subband length
    int inputOffset = computeThreadGroupOffset - padding;

    // Check if outcome index is lower than low subband length
    if(z >= lowLength) {
        return;
    }

    output[z * width * height + y * width + x] = forwardStepLow(sharedInput, inputOffset, z, analysisLowLength,
                                                                analysisLowFirstIndex);

    // Check if outcome index is lower than low subband length
    if(z + lowLength >= currFrames) {
        return;
    }

    output[(z + lowLength) * width * height + y * width + x] = forwardStepHigh(sharedInput, inputOffset, z,
                                                                               analysisHighLength, analysisHighFirstIndex);
}

__global__ void reverseTransform3DRow(float *input,
                                      float *output,
                                      int width,
                                      int height,
                                      int currWidth,
                                      int currHeight,
                                      int currFrames,
                                      int synthesisLowLength,
                                      int synthesisHighLength,
                                      int synthesisLowFirstIndex,
                                      int synthesisHighFirstIndex) {
    // Calculate filter padding
    int padding = getFilterPadding(synthesisLowLength, synthesisHighLength, synthesisLowFirstIndex, synthesisHighFirstIndex);
    int computeThreadCount = THREADS_PER_BLOCK - 2 * padding;
    int blockNumX = getBlockNum(currWidth, computeThreadCount);
    int threadComputeGroupOffset    = (blockIdx.x % blockNumX) * (THREADS_PER_BLOCK - 2 * padding);   // index of first compute thread
    int threadLoadGroupOffset       = threadComputeGroupOffset - padding;               // index of first load thread
    int loadThreadId                = threadLoadGroupOffset + threadIdx.x;              // current thread load element index
    int x = threadComputeGroupOffset + threadIdx.x - padding;                           // current thread compute element index
    int y = blockIdx.x / blockNumX;
    int z = blockIdx.y;

    bool analysisIgnoreEven = synthesisHighLength % 2 == 0; // Check wether ignore even or odd input values
    int lowLength           = (currWidth + 1) / 2;          // Low subband length
    int highLength          = currWidth - lowLength;        // High subband length
    int inputOffset         = threadComputeGroupOffset - padding;

    // Load input data to shared memory; Each thread one element for low pass and one element for high pass
    __shared__ float lowInput [THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];
    __shared__ float highInput[THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];

    lowInput[threadIdx.x]  = getInputValue3D(input, loadThreadId, y, z, FIXED_ROW, width, height, 2 * lowLength,
                                             currHeight, currFrames, 0, true);
    highInput[threadIdx.x] = getInputValue3D(input, loadThreadId, y, z, FIXED_ROW, width, height, 2 * highLength,
                                             currHeight, currFrames, lowLength, analysisIgnoreEven, !analysisIgnoreEven);

    __syncthreads();

    // Check if outcome index is lower than low subband length and if thread is compute thread
    // (not used only to load data into shared memory)
    if(x >= currWidth || threadIdx.x < padding || threadIdx.x >= THREADS_PER_BLOCK - padding) {
        return;
    }

    output[z * width * height + y * width + x] = reverseStep(lowInput, highInput, inputOffset, x, synthesisLowLength, synthesisHighLength,
                                                             synthesisLowFirstIndex, synthesisHighFirstIndex);
}

__global__ void reverseTransform3DColumn(float *input,
                                         float *output,
                                         int width,
                                         int height,
                                         int currWidth,
                                         int currHeight,
                                         int currFrames,
                                         int synthesisLowLength,
                                         int synthesisHighLength,
                                         int synthesisLowFirstIndex,
                                         int synthesisHighFirstIndex) {
    // Calculate filter padding
    int padding = getFilterPadding(synthesisLowLength, synthesisHighLength, synthesisLowFirstIndex, synthesisHighFirstIndex);
    int computeThreadCount = THREADS_PER_BLOCK - 2 * padding;
    int blockNumX = getBlockNum(currHeight, computeThreadCount);
    int threadComputeGroupOffset    = (blockIdx.x % blockNumX) * (THREADS_PER_BLOCK - 2 * padding);   // index of first compute thread
    int threadLoadGroupOffset       = threadComputeGroupOffset - padding;               // index of first load thread
    int loadThreadId                = threadLoadGroupOffset + threadIdx.x;              // current thread load element index
    int y = threadComputeGroupOffset + threadIdx.x - padding;                           // current thread compute element index
    int x = blockIdx.x / blockNumX;
    int z = blockIdx.y;

    bool analysisIgnoreEven = synthesisHighLength % 2 == 0; // Check wether ignore even or odd input values
    int lowLength           = (currHeight + 1) / 2;          // Low subband length
    int highLength          = currHeight - lowLength;        // High subband length
    int inputOffset         = threadComputeGroupOffset - padding;

    // Load input data to shared memory; Each thread one element for low pass and one element for high pass
    __shared__ float lowInput [THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];
    __shared__ float highInput[THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];

    lowInput[threadIdx.x]  = getInputValue3D(input, x, loadThreadId, z, FIXED_COLUMN, width, height, currWidth,
                                             2 * lowLength, currFrames, 0, true);
    highInput[threadIdx.x] = getInputValue3D(input, x, loadThreadId, z, FIXED_COLUMN, width, height, currWidth,
                                             2 * highLength, currFrames, lowLength, analysisIgnoreEven, !analysisIgnoreEven);

    __syncthreads();

    // Check if outcome index is lower than low subband length and if thread is compute thread
    // (not used only to load data into shared memory)
    if(y >= currHeight || threadIdx.x < padding || threadIdx.x >= THREADS_PER_BLOCK - padding) {
        return;
    }

    output[z * width * height + y * width + x] = reverseStep(lowInput, highInput, inputOffset, y, synthesisLowLength, synthesisHighLength,
                                                             synthesisLowFirstIndex, synthesisHighFirstIndex);
}

__global__ void reverseTransform3DFrame(float *input,
                                        float *output,
                                        int width,
                                        int height,
                                        int currWidth,
                                        int currHeight,
                                        int currFrames,
                                        int synthesisLowLength,
                                        int synthesisHighLength,
                                        int synthesisLowFirstIndex,
                                        int synthesisHighFirstIndex) {
    // Calculate filter padding
    int padding = getFilterPadding(synthesisLowLength, synthesisHighLength, synthesisLowFirstIndex, synthesisHighFirstIndex);
    int computeThreadCount = THREADS_PER_BLOCK - 2 * padding;
    int blockNumX = getBlockNum(currFrames, computeThreadCount);
    int threadComputeGroupOffset    = (blockIdx.x % blockNumX) * (THREADS_PER_BLOCK - 2 * padding);   // index of first compute thread
    int threadLoadGroupOffset       = threadComputeGroupOffset - padding;               // index of first load thread
    int loadThreadId                = threadLoadGroupOffset + threadIdx.x;              // current thread load element index
    int z = threadComputeGroupOffset + threadIdx.x - padding;                           // current thread compute element index
    int x = blockIdx.x / blockNumX;
    int y = blockIdx.y;

    bool analysisIgnoreEven = synthesisHighLength % 2 == 0; // Check wether ignore even or odd input values
    int lowLength           = (currFrames + 1) / 2;          // Low subband length
    int highLength          = currFrames - lowLength;        // High subband length
    int inputOffset         = threadComputeGroupOffset - padding;

    // Load input data to shared memory; Each thread one element for low pass and one element for high pass
    __shared__ float lowInput [THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];
    __shared__ float highInput[THREADS_PER_BLOCK + 2 * FILTER_BUFF_LENGTH];

    lowInput[threadIdx.x]  = getInputValue3D(input, x, y, loadThreadId, FIXED_FRAME, width, height, currWidth,
                                             currHeight, 2 * lowLength, 0, true);
    highInput[threadIdx.x] = getInputValue3D(input, x, y, loadThreadId, FIXED_FRAME, width, height, currWidth,
                                             currHeight, 2 * highLength, lowLength, analysisIgnoreEven, !analysisIgnoreEven);

    __syncthreads();

    // Check if outcome index is lower than low subband length and if thread is compute thread
    // (not used only to load data into shared memory)
    if(z >= currFrames || threadIdx.x < padding || threadIdx.x >= THREADS_PER_BLOCK - padding) {
        return;
    }

    output[z * width * height + y * width + x] = reverseStep(lowInput, highInput, inputOffset, z, synthesisLowLength, synthesisHighLength,
                                                             synthesisLowFirstIndex, synthesisHighFirstIndex);
}
