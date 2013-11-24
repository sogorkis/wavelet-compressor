/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "WaveletTransformImpl.h"
#include "util/WaveletCompressorUtil.h"

#include <cstring>
#include <cstdio>
#include <algorithm>

//#define DEBUG_TRANSFORM

void WaveletTransformImpl::forward1D(const float *input, float *output, int length, int levels, Wavelet *wavelet) {
    float * buff = new float[length];

    int currLength = length;
    memcpy(buff, input, sizeof(float) * currLength);
    while(levels-- > 0) {
        if(currLength < 3 && wavelet->isSymmetric()) {
            fail("Low pass subband to small, reduce levels.");
        }

        transformForward(buff, output, currLength, wavelet);

        currLength = (currLength+1) / 2;

        memcpy(buff, output, sizeof(float) * currLength);
    }

    delete buff;
}

void WaveletTransformImpl::reverse1D(const float *input, float *output, int length, int levels, Wavelet *wavelet) {
    float * buff = new float[length];
    int currLength[levels];

    currLength[0] = length;
    for(int i = 1; i < levels; ++i) {
        currLength[i] = (currLength[i - 1] + 1) / 2;
    }

    memcpy(buff, input, sizeof(float) * length);
    while(levels-- > 0) {
        transformReverse(buff, output, currLength[levels], wavelet);

        memcpy(buff, output, sizeof(float) * currLength[levels]);
    }

    delete buff;
}

void WaveletTransformImpl::forward2D(const float *input, float *output, int width, int height, int levels, Wavelet *wavelet) {
    float * inputBuff   = new float[std::max(width, height)];
    float * outputBuff  = new float[std::max(width, height)];

    int currWidth = width;
    int currHeight = height;
    memcpy(output, input, width * height * sizeof(float));
    while(levels-- > 0) {
        if((currWidth < 3 || currHeight < 3) && wavelet->isSymmetric()) {
            fail("Low pass subband to small, reduce levels.");
        }

        // Apply forward transform for each row
        for(int i = 0; i < currHeight; ++i) {
            // Copy input row data to buff
            copyRowFrom2D(output, inputBuff, i, currWidth, width);

            // Apply transform to row
            transformForward(inputBuff, outputBuff, currWidth, wavelet);

            //Copy transformed row to output
            copyRowTo2D(outputBuff, output, i, currWidth, width);
        }

        // Apply forward transform for each column
        for(int i = 0; i < currWidth; ++i) {
            // Copy input column data to buff
            copyColumnFrom2D(output, inputBuff, i, width, currHeight);

            // Apply transform to row
            transformForward(inputBuff, outputBuff, currHeight, wavelet);

            //Copy transformed column to output
            copyColumnTo2D(outputBuff, output, i, width, currHeight);
        }

        currWidth = (currWidth+1) / 2;
        currHeight = (currHeight+1) / 2;
    }

    delete inputBuff;
    delete outputBuff;
}

void WaveletTransformImpl::reverse2D(const float *input, float *output, int width, int height, int levels, Wavelet *wavelet) {
    float * inputBuff   = new float[std::max(width, height)];
    float * outputBuff  = new float[std::max(width, height)];
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

    memcpy(output, input, width * height * sizeof(float));
    while(levels-- > 0) {

        // Apply reverse transform for each row
        for(int i = 0; i < currHeight[levels]; ++i) {
            // Copy input row data to buff
            copyRowFrom2D(output, inputBuff, i, currWidth[levels], width);

            // Apply transform to row
            transformReverse(inputBuff, outputBuff, currWidth[levels], wavelet);

            //Copy transformed row to output
            copyRowTo2D(outputBuff, output, i, currWidth[levels], width);
        }

        // Apply reverse transform for each column
        for(int i = 0; i < currWidth[levels]; ++i) {

            // Copy input row data to buff
            copyColumnFrom2D(output, inputBuff, i, width, currHeight[levels]);

            // Apply transform to row
            transformReverse(inputBuff, outputBuff, currHeight[levels], wavelet);

            //Copy transformed row to output
            copyColumnTo2D(outputBuff, output, i, width, currHeight[levels]);
        }
    }

    delete inputBuff;
    delete outputBuff;
}

void WaveletTransformImpl::forward3D(const float *input, float *output, int width, int height, int frames, int levels, Wavelet *wavelet) {
    float * inputBuff = new float[std::max(std::max(width, height), frames)];
    float * outputBuff = new float[std::max(std::max(width, height), frames)];

    int currWidth = width;
    int currHeight = height;
    int currDepth = frames;
    memcpy(output, input, width * height * frames * sizeof(float));
    while(levels-- > 0) {
        if((currWidth < 3 || currHeight < 3 || currDepth < 3) && wavelet->isSymmetric()) {
            fail("Low pass subband to small, reduce levels.");
        }

        // Apply forward transform for each row
        for(int j = 0; j < currDepth; ++j) {
            for(int i = 0; i < currHeight; ++i) {
                // Copy input row data to buff
                copyRowFrom3D(output, inputBuff, i, j, currWidth, width, height);

                // Apply transform to row
                transformForward(inputBuff, outputBuff, currWidth, wavelet);

                //Copy transformed row to output
                copyRowTo3D(outputBuff, output, i, j, currWidth, width, height);
            }
        }

        // Apply forward transform for each column
        for(int j = 0; j < currDepth; ++j) {
            for(int i = 0; i < currWidth; ++i) {
                // Copy input column data to buff
                copyColumnFrom3D(output, inputBuff, i, j, currHeight, width, height);

                // Apply transform to row
                transformForward(inputBuff, outputBuff, currHeight, wavelet);

                //Copy transformed column to output
                copyColumnTo3D(outputBuff, output, i, j, currHeight, width, height);
            }
        }

        // Apply forward transform for each depth element
        for(int j = 0; j < currWidth; ++j) {
            for(int i = 0; i < currHeight; ++i) {
                // Copy input depths data to buff
                copyDepthsFrom3D(output, inputBuff, i, j, currDepth, width, height);

                // Apply transform to depth
                transformForward(inputBuff, outputBuff, currDepth, wavelet);

                //Copy transformed depths to output
                copyDepthsTo3D(outputBuff, output, i, j, currDepth, width, height);
            }
        }

        currWidth   = (currWidth+1) / 2;
        currHeight  = (currHeight+1) / 2;
        currDepth   = (currDepth+1) / 2;
    }

    delete inputBuff;
    delete outputBuff;
}

void WaveletTransformImpl::reverse3D(const float *input, float *output, int width, int height, int frames, int levels, Wavelet *wavelet) {
    float * inputBuff = new float[std::max(std::max(width, height), frames)];
    float * outputBuff = new float[std::max(std::max(width, height), frames)];
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

    memcpy(output, input, width * height * frames * sizeof(float));
    while(levels-- > 0) {

        // Apply reverse transform for each row
        for(int j = 0; j < currDepth[levels]; ++j) {
            for(int i = 0; i < currHeight[levels]; ++i) {
                // Copy input row data to buff
                copyRowFrom3D(output, inputBuff, i, j, currWidth[levels], width, height);

                // Apply transform to row
                transformReverse(inputBuff, outputBuff, currWidth[levels], wavelet);

                //Copy transformed row to output
                copyRowTo3D(outputBuff, output, i, j, currWidth[levels], width, height);
            }
        }

        // Apply reverse transform for each column
        for(int j = 0; j < currDepth[levels]; ++j) {
            for(int i = 0; i < currWidth[levels]; ++i) {
                // Copy input column data to buff
                copyColumnFrom3D(output, inputBuff, i, j, currHeight[levels], width, height);

                // Apply transform to row
                transformReverse(inputBuff, outputBuff, currHeight[levels], wavelet);

                //Copy transformed column to output
                copyColumnTo3D(outputBuff, output, i, j, currHeight[levels], width, height);
            }
        }

        // Apply reverse transform for each depth element
        for(int j = 0; j < currWidth[levels]; ++j) {
            for(int i = 0; i < currHeight[levels]; ++i) {
                // Copy input depths data to buff
                copyDepthsFrom3D(output, inputBuff, i, j, currDepth[levels], width, height);

                // Apply transform to depths
                transformReverse(inputBuff, outputBuff, currDepth[levels], wavelet);

                //Copy transformed depths to output
                copyDepthsTo3D(outputBuff, output, i, j, currDepth[levels], width, height);
            }
        }
    }

    delete inputBuff;
    delete outputBuff;
}

void WaveletTransformImpl::linearize2D(const float *input, float *output, int width, int height, int levels) {
    int subbandCount = getSubbandCount2D(levels);
    int k = 0;

    for(int subbandIndex = 0; subbandIndex < subbandCount; ++subbandIndex) {
        int subbandWidth    = getSubbandWidth2D(subbandIndex, subbandCount, width);
        int subbandHeight   = getSubbandHeight2D(subbandIndex, subbandCount, height);
        int subbandXOffset  = getSubbandXOffset2D(subbandIndex, subbandCount, width);
        int subbandYOffset  = getSubbandYOffset2D(subbandIndex, subbandCount, height);

        for (int y = subbandYOffset; y < subbandYOffset + subbandHeight; ++y) {
            for (int x = subbandXOffset; x < subbandXOffset + subbandWidth; ++x) {
                output[k++] = input[y * width + x];
            }
        }
    }
}

void WaveletTransformImpl::delinearize2D(const float *input, float *output, int width, int height, int levels) {
    int subbandCount = getSubbandCount2D(levels);
    int k = 0;

    for(int subbandIndex = 0; subbandIndex < subbandCount; ++subbandIndex) {
        int subbandWidth    = getSubbandWidth2D(subbandIndex, subbandCount, width);
        int subbandHeight   = getSubbandHeight2D(subbandIndex, subbandCount, height);
        int subbandXOffset  = getSubbandXOffset2D(subbandIndex, subbandCount, width);
        int subbandYOffset  = getSubbandYOffset2D(subbandIndex, subbandCount, height);

        for (int y = subbandYOffset; y < subbandYOffset + subbandHeight; ++y) {
            for (int x = subbandXOffset; x < subbandXOffset + subbandWidth; ++x) {
                output[y * width + x] = input[k++];
            }
        }
    }
}

void WaveletTransformImpl::linearize3D(const float *input, float *output, int width, int height, int frames, int levels) {
    int subbandCount = getSubbandCount3D(levels);
    int k = 0;

    for(int subbandIndex = 0; subbandIndex < subbandCount; ++subbandIndex) {
        int subbandWidth    = getSubbandWidth3D(subbandIndex, subbandCount, width);
        int subbandHeight   = getSubbandHeight3D(subbandIndex, subbandCount, height);
        int subbandFrames   = getSubbandFrames3D(subbandIndex, subbandCount, frames);
        int subbandXOffset  = getSubbandXOffset3D(subbandIndex, subbandCount, width);
        int subbandYOffset  = getSubbandYOffset3D(subbandIndex, subbandCount, height);
        int subbandZOffset  = getSubbandZOffset3D(subbandIndex, subbandCount, frames);

        debug("subband %2d, W=%3d, H=%3d, F=%3d, X=%3d, Y=%3d, Z=%3d", subbandIndex, subbandWidth, subbandHeight,
             subbandFrames, subbandXOffset, subbandYOffset, subbandZOffset);

        for (int z = subbandZOffset; z < subbandZOffset + subbandFrames; ++z) {
            for (int y = subbandYOffset; y < subbandYOffset + subbandHeight; ++y) {
                for (int x = subbandXOffset; x < subbandXOffset + subbandWidth; ++x) {
                    float l = input[z * width * height + y * width + x];
                    output[k++] = l;
                }
            }
        }
    }
}

void WaveletTransformImpl::delinearize3D(const float *input, float *output, int width, int height, int frames, int levels) {
    int subbandCount = getSubbandCount3D(levels);
    int k = 0;

    for(int subbandIndex = 0; subbandIndex < subbandCount; ++subbandIndex) {
        int subbandWidth    = getSubbandWidth3D(subbandIndex, subbandCount, width);
        int subbandHeight   = getSubbandHeight3D(subbandIndex, subbandCount, height);
        int subbandFrames   = getSubbandFrames3D(subbandIndex, subbandCount, frames);
        int subbandXOffset  = getSubbandXOffset3D(subbandIndex, subbandCount, width);
        int subbandYOffset  = getSubbandYOffset3D(subbandIndex, subbandCount, height);
        int subbandZOffset  = getSubbandZOffset3D(subbandIndex, subbandCount, frames);

        for (int z = subbandZOffset; z < subbandZOffset + subbandFrames; ++z) {
            for (int y = subbandYOffset; y < subbandYOffset + subbandHeight; ++y) {
                for (int x = subbandXOffset; x < subbandXOffset + subbandWidth; ++x) {
                    output[z * width * height + y * width + x] = input[k++];
                }
            }
        }
    }
}

void WaveletTransformImpl::transformForward(const float *input, float *output, int length, Wavelet *wavelet) {
    int lowLength = (length + 1) / 2;
    int analysisLowLength       = wavelet->getAnalysisLowFilter()->getLength();
    int analysisHighLength      = wavelet->getAnalysisHighFilter()->getLength();
    int analysisLowFirstIndex   = wavelet->getAnalysisLowFilter()->getFirstIndex();
    int analysisHighFirstIndex  = wavelet->getAnalysisHighFilter()->getFirstIndex();
    const float * analysisLow   = wavelet->getAnalysisLowFilter()->getValues();
    const float * analysisHigh  = wavelet->getAnalysisHighFilter()->getValues();

    memset(output, 0, sizeof(float) * length);

    // Calculate convolution for low and high pass analysis filters at one time
    for(int i = 0; i < lowLength; ++i) {

        // Convolve with low pass analysis filter
        for(int j = 0; j < analysisLowLength; ++j) {
            int k = (2 * i) + j - analysisLowFirstIndex;
            float inputValue = getInputValue(input, k, length);
            if(inputValue == 0) {
                continue;
            }

            output[i] += inputValue * analysisLow[j];

#ifdef DEBUG_TRANSFORM
            printf ("output[%2d] += input[%2d](%10f) * AL[%2d](%10f)\n",
                    i, k, inputValue, j, analysisLow[j]);
#endif
        }

        // Convolve with high pass analysis filter
        for(int j = 0; j < analysisHighLength; ++j) {
            if(i + lowLength >= length) {
                break; // this happen with odd input length
            }

            int k = (2 * i) + j - analysisHighFirstIndex;
            float inputValue = getInputValue(input, k, length);
            if(inputValue == 0) {
                continue;
            }

            output[i + lowLength] += inputValue * analysisHigh[j];

#ifdef DEBUG_TRANSFORM
            printf ("output[%2d] += input[%2d](%10f) * AH[%2d](%10f)\n",
                    i + lowLength, k, inputValue, j, analysisHigh[j]);
#endif
        }
    }
}

void WaveletTransformImpl::transformReverse(const float *input, float *output, int length, Wavelet *wavelet) {
    int lowLength = (length + 1) / 2;
    int highLength = length - lowLength;
    int synthesisLowLength      = wavelet->getSynthesisLowFilter()->getLength();
    int synthesisHighLength     = wavelet->getSynthesisHighFilter()->getLength();
    int synthesisLowFirstIndex  = wavelet->getSynthesisLowFilter()->getFirstIndex();
    int synthesisHighFirstIndex = wavelet->getSynthesisHighFilter()->getFirstIndex();
    const float * synthesisLow  = wavelet->getSynthesisLowFilter()->getValues();
    const float * synthesisHigh = wavelet->getSynthesisHighFilter()->getValues();
    bool analysisIgnoreEven     = synthesisHighLength % 2 == 0;

    // Calculate convolution for low and high pass synthesis filters
    for(int i = 0; i < length; ++i) {
        float value = 0.0f;

        // Convolve with low pass synthesis filter
        for(int j = 0; j < synthesisLowLength; ++j) {
            int k = i - j + synthesisLowFirstIndex;
            float inputValue = getInputValue(input, k, 2 * lowLength, 0, true, false, false);
            if(inputValue == 0) {
                continue;
            }

            value += inputValue * synthesisLow[j];

#ifdef DEBUG_TRANSFORM
            printf ("output[%2d](%10f) += input[%2d](%10f) * SL[%2d](%10f)\n",
                    i, value, k/2, inputValue, j, synthesisLow[j]);
#endif
        }

        // Convolve with high pass synthesis filter
        for(int j = 0; j < synthesisHighLength; ++j) {
            int k = i - j + synthesisHighFirstIndex;
            float inputValue = getInputValue(input, k, 2 * highLength, lowLength, analysisIgnoreEven, !analysisIgnoreEven, true);
            if(inputValue == 0) {
                continue;
            }

            value += inputValue * synthesisHigh[j];

#ifdef DEBUG_TRANSFORM
            printf ("output[%2d](%10f) += input[%2d](%10f) * SH[%2d](%10f)\n",
                    i, value, k/2+lowLength, inputValue, j, synthesisHigh[j]);
#endif
        }

        output[i] = value;
    }
}

inline float WaveletTransformImpl::getInputValue(const float *input, int index, int inputLength, int offset,
                                                 bool ignoreOddIndex, bool ignoreEvenIndex, bool asymmetric) {
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
                newIndex = index - inputLength;
            }
        }

    }

    if(ignoreOddIndex) {
        if(abs(newIndex) % 2 == 1) {
            return 0;
        }
        return input[newIndex/2 + offset];
    }
    else if(ignoreEvenIndex) {
        if(abs(newIndex) % 2 == 0) {
            return 0;
        }
        return input[newIndex/2 + offset];
    }
    return input[newIndex + offset];
}

void WaveletTransformImpl::copyRowFrom2D(const float * input, float * output, int rowIndex, int length, int width) {
    int offset = rowIndex * width;
    memcpy(output, input + offset, length * sizeof(float));
}

void WaveletTransformImpl::copyRowTo2D(const float * input, float * output, int rowIndex, int length, int width) {
    int offset = rowIndex * width;
    memcpy(output + offset, input, length * sizeof(float));
}


void WaveletTransformImpl::copyColumnFrom2D(const float * input, float * output, int columnIndex, int width, int height) {
    for(int i = 0; i < height; ++i) {
        output[i] = input[i * width + columnIndex];
    }
}

void WaveletTransformImpl::copyColumnTo2D(const float * input, float * output, int columnIndex, int width, int height) {
    for(int i = 0; i < height; ++i) {
        output[i * width + columnIndex] = input[i];
    }
}

void WaveletTransformImpl::copyRowFrom3D(const float *input, float *output, int rowIndex, int depthIndex, int length, int width, int height) {
    int offset = depthIndex * width * height;
    copyRowFrom2D(input + offset, output, rowIndex, length, width);
}

void WaveletTransformImpl::copyRowTo3D(const float *input, float *output, int rowIndex, int depthIndex, int length, int width, int height) {
    int offset = depthIndex * width * height;
    copyRowTo2D(input, output + offset, rowIndex, length, width);
}

void WaveletTransformImpl::copyColumnFrom3D(const float *input, float *output, int columnIndex, int depthIndex, int length, int width, int height) {
    int offset = depthIndex * width * height;
    copyColumnFrom2D(input + offset, output, columnIndex, width, length);
}

void WaveletTransformImpl::copyColumnTo3D(const float *input, float *output, int columnIndex, int depthIndex, int length, int width, int height) {
    int offset = depthIndex * width * height;
    copyColumnTo2D(input, output + offset, columnIndex, width, length);
}

void WaveletTransformImpl::copyDepthsFrom3D(const float *input, float *output, int rowIndex, int columnIndex, int length, int width, int height) {
    for(int i = 0; i < length; ++i) {
        output[i] = input[i * width * height + rowIndex * width + columnIndex];
    }
}

void WaveletTransformImpl::copyDepthsTo3D(const float *input, float *output, int rowIndex, int columnIndex, int length, int width, int height) {
    for(int i = 0; i < length; ++i) {
        output[i * width * height + rowIndex * width + columnIndex] = input[i];
    }
}
