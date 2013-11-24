/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "CudaWaveletCompressor.h"
#include "wavelet_transform/WaveletTransformCudaImpl.h"
#include "color_transform/YCbCrColorTransformCuda.h"
#include "wavelet_transform/WaveletFactory.h"
#include "util/WaveletCompressorUtil.h"
#include "util/TimersData.h"

#include <cutil_inline.h>

void CudaWaveletCompressor::encode(Image *image,
                                   Wavelet *wavelet,
                                   float targetRate,
                                   std::ostream *outStream,
                                   int levels,
                                   QuantizerType quantizerType,
                                   ArithCoderModelType arithCoderModelType,
                                   ColorTransformType colorTransformType) {
    WaveletTransformCudaImpl cudaTransform;
    ArithCoder arithCoder;
    ArithCoderModel *quantArithModel = ArithCoderModelFactory::getInstance(&arithCoder, arithCoderModelType, 8);
    int width = image->getWidth(), height = image->getHeight();
    int channelCount    = image->getChannelCount();
    int imageSize       = height * width;
    int subbandCount    = getSubbandCount2D(levels);

    info("Encoding image - size: %dx%d, channels: %d", width, height, channelCount);
    info("wavelet: %s, decompositionLevelsCount: %d", wavelet->getName().c_str(), levels);

    writeImageHeader(outStream, image, wavelet, levels, quantizerType, arithCoderModelType, colorTransformType);

    arithCoder.SetOutputFile(outStream);

    cudaSetDevice(cutGetMaxGflopsDeviceId());

    float * deviceInput[channelCount];
    float * deviceOutput;
    float * firstBuff = new float[width * height];
    float * secondBuff = new float[width * height];

    // Allocate device memory
    TimersData::getInstance()->startTimer("cudaMalloc");
    cutilSafeCall(cudaMalloc((void**)&deviceOutput, width * height * sizeof(float)));
    for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
        cutilSafeCall(cudaMalloc((void**)&deviceInput[channelIndex], width * height * sizeof(float)));
    }
    TimersData::getInstance()->stopTimer("cudaMalloc");;

    // Copy input data
    TimersData::getInstance()->startTimer("cudaHostDevice");
    for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
        cutilSafeCall(cudaMemcpy(deviceInput[channelIndex], image->getChannelData(channelIndex),
                                 width * height * sizeof(float), cudaMemcpyHostToDevice));
    }
    TimersData::getInstance()->stopTimer("cudaHostDevice");

    TimersData::getInstance()->startTimer("cudaColorTransform");
    ColorTransform *colorTransform = NULL;
    // Perform color transform if necessary
    if (colorTransformType == YCBCR_COLOR_TRANSFORM) {
        colorTransform = new YCbCrColorTransformCuda();
        ((YCbCrColorTransformCuda *) colorTransform)->transformForwardDevice(deviceInput, width * height);
    }
    else {
        colorTransform = new EmptyColorTransform();
    }
    TimersData::getInstance()->stopTimer("cudaColorTransform");

    for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
        TimersData::getInstance()->startTimer("cudaTransform");
        cudaTransform.forwardDeviceMemory2D(deviceOutput, deviceInput[channelIndex], width, height, levels, wavelet);
        TimersData::getInstance()->stopTimer("cudaTransform");

        TimersData::getInstance()->startTimer("cudaDeviceHost");
        cutilSafeCall(cudaMemcpy(firstBuff, deviceInput[channelIndex], width * height * sizeof(float), cudaMemcpyDeviceToHost));
        TimersData::getInstance()->stopTimer("cudaDeviceHost");

        TimersData::getInstance()->startTimer("linearize");
        cudaTransform.linearize2D(firstBuff, secondBuff, width, height, levels);
        TimersData::getInstance()->stopTimer("linearize");

        int byteBudget = (channelCount * imageSize * colorTransform->getChannelBudgetRatio(channelIndex)) / targetRate;
        ImageSubbandData subbandData(secondBuff, subbandCount, width, height);

        quantizeAndEncode(subbandData, subbandCount, quantizerType, arithCoderModelType,
                          arithCoder, quantArithModel, byteBudget, channelIndex);
    }

    quantArithModel->FinishEncode();

    for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
        cudaFree(deviceInput[channelIndex]);
    }
    cudaFree(deviceOutput);
    delete [] firstBuff;
    delete colorTransform;
    delete quantArithModel;

    cudaThreadExit();
}

void CudaWaveletCompressor::encode(ImageSequence *imageSequence,
                                   Wavelet *wavelet,
                                   float targetRate,
                                   std::ostream *outStream,
                                   int levels,
                                   QuantizerType quantizerType,
                                   ArithCoderModelType arithCoderModelType,
                                   ColorTransformType colorTransformType) {
    WaveletTransformCudaImpl transform;
    ArithCoder arithCoder;
    ArithCoderModel *quantArithModel = ArithCoderModelFactory::getInstance(&arithCoder, arithCoderModelType, 8);
    int width = imageSequence->getWidth(), height = imageSequence->getHeight();
    int channelCount    = imageSequence->getChannelCount();
    int frameCount      = imageSequence->getFrameCount();
    int frameGroupSize  = 64;
    int frameGroupCount = (frameCount - 1) / frameGroupSize + 1;
    int subbandCount    = getSubbandCount3D(levels);
    float **hostInput   = new float*[channelCount];
    float **deviceInput = new float*[channelCount];
    float *hostOutput, *deviceOutput;

    // Allocate memory
    TimersData::getInstance()->startTimer("cudaMalloc");
    hostOutput = new float[width * height * frameGroupSize];
    cutilSafeCall(cudaMalloc((void**)&deviceOutput, width * height * frameGroupSize * sizeof(float)));
    for(int i = 0; i < channelCount; ++i) {
        hostInput[i] = new float[width * height * frameGroupSize];
        cutilSafeCall(cudaMalloc((void**)&deviceInput[i], width * height * frameGroupSize * sizeof(float)));
    }
    TimersData::getInstance()->stopTimer("cudaMalloc");;

    writeImageSequenceHeader(outStream, imageSequence, wavelet, levels, quantizerType, arithCoderModelType,
                             colorTransformType);

    arithCoder.SetOutputFile(outStream);

    for(int frameGroupIndex = 0; frameGroupIndex < frameGroupCount; ++frameGroupIndex) {
        // Load frames into host memory
        TimersData::getInstance()->startTimer("openImageSequence");
        int readFrames = imageSequence->readImages(hostInput, frameGroupSize);
        TimersData::getInstance()->stopTimer("openImageSequence");

        TimersData::getInstance()->startTimer("cudaHostDevice");
        // Load frames into device memory
        for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
            cutilSafeCall(cudaMemcpy(deviceInput[channelIndex], hostInput[channelIndex],
                                     width * height * readFrames * sizeof(float), cudaMemcpyHostToDevice));
        }
        TimersData::getInstance()->stopTimer("cudaHostDevice");

        TimersData::getInstance()->startTimer("cudaColorTransform");
        ColorTransform *colorTransform = NULL;
        // Perform color transform if necessary
        if (colorTransformType == YCBCR_COLOR_TRANSFORM) {
            colorTransform = new YCbCrColorTransformCuda();
            ((YCbCrColorTransformCuda *) colorTransform)->transformForwardDevice(deviceInput, width * height * readFrames);
        }
        else {
            colorTransform = new EmptyColorTransform();
        }
        TimersData::getInstance()->stopTimer("cudaColorTransform");

        for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
            TimersData::getInstance()->startTimer("cudaTransform");
            transform.forwardDeviceMemory3D(deviceInput[channelIndex], deviceOutput, width, height, readFrames,
                                            levels, wavelet);
            TimersData::getInstance()->stopTimer("cudaTransform");

            TimersData::getInstance()->startTimer("cudaDeviceHost");
            cutilSafeCall(cudaMemcpy(hostOutput, deviceOutput, width * height * readFrames * sizeof(float),
                                     cudaMemcpyDeviceToHost));
            TimersData::getInstance()->stopTimer("cudaDeviceHost");

            TimersData::getInstance()->startTimer("linearize");
            transform.linearize3D(hostOutput, hostInput[channelIndex], width, height, readFrames, levels);
            TimersData::getInstance()->stopTimer("linearize");

            int byteBudget = (width * height * readFrames * channelCount
                              * colorTransform->getChannelBudgetRatio(channelIndex)) / targetRate;
            ImageSequenceSubbandData subbandData(hostInput[channelIndex], subbandCount, width, height, readFrames);

            quantizeAndEncode(subbandData, subbandCount, quantizerType, arithCoderModelType,
                              arithCoder, quantArithModel, byteBudget, channelIndex);
        }

        delete colorTransform;
    }

    quantArithModel->FinishEncode();

    for(int i = 0; i < channelCount; ++i) {
        delete [] hostInput[i];
        cudaFree(deviceInput[i]);
    }
    cudaFree(deviceOutput);
    delete [] deviceInput;
    delete [] hostInput;
    delete quantArithModel;

    cudaThreadExit();
}

void CudaWaveletCompressor::decode(std::istream *inStream,
                                   Image *image) {
    int width, height, channelCount, levels;
    std::string waveletName;
    QuantizerType quantizerType;
    ArithCoderModelType arithCoderModelType;
    ColorTransformType colorTransformType;

    readImageHeader(inStream, width, height, channelCount, levels, waveletName, quantizerType,
                    arithCoderModelType, colorTransformType);
    Wavelet * wavelet = WaveletFactory::getInstance(waveletName);
    if(wavelet == NULL) {
        fail("Invalid wavelet name %s", waveletName.c_str());
    }
    info("Decoding image - size: %dx%d, channels: %d", width, height, channelCount);
    info("wavelet: %s, decompositionLevelsCount: %d", waveletName.c_str(), levels);

    image->init(width, height, channelCount);
    int subbandCount = getSubbandCount2D(levels);
    WaveletTransformCudaImpl cudaTransform;
    ArithCoder arithCoder;
    ArithCoderModel *quantArithModel = ArithCoderModelFactory::getInstance(&arithCoder, arithCoderModelType, 8);
    ColorTransform *colorTransform = ColorTransformFactory::getInstance(colorTransformType);
    float *firstBuff    = new float[height * width];
    float *secondBuff   = new float[height * width];

    float *deviceInput;
    float *deviceOutput[channelCount];

    TimersData::getInstance()->startTimer("cudaMalloc");
    cutilSafeCall(cudaMalloc((void**) &deviceInput, width * height * sizeof(float)));
    for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
        cutilSafeCall(cudaMalloc((void**) &deviceOutput[channelIndex], width * height * sizeof(float)));
    }
    TimersData::getInstance()->stopTimer("cudaMalloc");

    arithCoder.SetInputFile(inStream);
    arithCoder.DecodeStart();

    for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
        TimersData::getInstance()->startTimer("dequantizeDecode");
        ImageSubbandData subbandData(firstBuff, subbandCount, width, height);

        decodeAndDequantize(subbandData, subbandCount, quantizerType, arithCoderModelType, arithCoder, quantArithModel);
        TimersData::getInstance()->stopTimer("dequantizeDecode");

        TimersData::getInstance()->startTimer("delinearize");
        cudaTransform.delinearize2D(firstBuff, secondBuff, width, height, levels);
        TimersData::getInstance()->stopTimer("delinearize");

        TimersData::getInstance()->startTimer("cudaHostDevice");
        cutilSafeCall(cudaMemcpy(deviceInput, secondBuff, width * height * sizeof(float), cudaMemcpyHostToDevice));
        cutilSafeCall(cudaMemcpy(deviceOutput[channelIndex], secondBuff, width * height * sizeof(float), cudaMemcpyHostToDevice));
        TimersData::getInstance()->stopTimer("cudaHostDevice");

        TimersData::getInstance()->startTimer("cudaTransform");
        cudaTransform.reverseDeviceMemory2D(deviceInput, deviceOutput[channelIndex], width, height, levels, wavelet);
        TimersData::getInstance()->stopTimer("cudaTransform");
    }

    TimersData::getInstance()->startTimer("cudaColorTransform");
    // Perform reverese color transform if necessary
    if (colorTransformType == YCBCR_COLOR_TRANSFORM) {
        YCbCrColorTransformCuda colorTransform;
        colorTransform.transformReverseDevice(deviceOutput, width * height);
    }
    TimersData::getInstance()->startTimer("cudaColorStop");

    TimersData::getInstance()->startTimer("cudaDeviceHost");
    for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
        cutilSafeCall(cudaMemcpy(image->getChannelData(channelIndex), deviceOutput[channelIndex],
                                 width * height * sizeof(float), cudaMemcpyDeviceToHost));
    }
    TimersData::getInstance()->stopTimer("cudaDeviceHost");

    cudaFree(deviceInput);
    for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
        cudaFree(deviceOutput[channelIndex]);
    }
    delete wavelet;
    delete colorTransform;
    delete quantArithModel;
    delete [] firstBuff;
    delete [] secondBuff;

    cudaThreadExit();
}

void CudaWaveletCompressor::decode(std::istream *inStream,
                                   ImageSequence *imageSequence) {
    long frames;
    int fps, width, height, channelCount, levles;
    std::string waveletName;
    QuantizerType quantizerType;
    ArithCoderModelType arithCoderModelType;
    ColorTransformType colorTransformType;

    readImageSequenceHeader(inStream, frames, fps, width, height, channelCount, levles, waveletName,
                            quantizerType, arithCoderModelType, colorTransformType);

    Wavelet * wavelet = WaveletFactory::getInstance(waveletName);
    if(wavelet == NULL) {
        fail("Invalid wavelet name %s", waveletName.c_str());
    }
    info("Decoding image sequence - frames: %d, fps %d, size: %dx%d, channels: %d", frames, fps,
         width, height, channelCount);
    info("wavelet: %s, decompositionLevelsCount: %d", waveletName.c_str(), levles);

    int subbandCount = getSubbandCount3D(levles);
    WaveletTransformCudaImpl transform;
    ArithCoder arithCoder;
    ArithCoderModel *quantArithModel = ArithCoderModelFactory::getInstance(&arithCoder, arithCoderModelType, 8);
    int frameGroupSize  = 64;
    int frameGroupCount = (frames - 1) / frameGroupSize + 1;
    float *dequantizedData = new float[width * height * frameGroupSize];

    float *deviceInput;
    float *deviceOutput[channelCount];

    TimersData::getInstance()->startTimer("cudaMalloc");
    cutilSafeCall(cudaMalloc((void**) &deviceInput, width * height * frameGroupSize * sizeof(float)));
    for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
        cutilSafeCall(cudaMalloc((void**) &deviceOutput[channelIndex], width * height * frameGroupSize * sizeof(float)));
    }
    TimersData::getInstance()->stopTimer("cudaMalloc");

    imageSequence->initWriteSequence(fps, width, height, frames, frameGroupSize, channelCount);

    arithCoder.SetInputFile(inStream);
    arithCoder.DecodeStart();

    for(int frameGroupIndex = 0; frameGroupIndex < frameGroupCount; ++frameGroupIndex) {
        int readFrames = frameGroupSize;
        if(frameGroupIndex == frameGroupCount - 1 && readFrames != frameGroupSize) {
            readFrames = frames % frameGroupSize;
        }

        for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
            TimersData::getInstance()->startTimer("dequantizeDecode");
            ImageSequenceSubbandData subbandData(imageSequence->getChannelData(channelIndex), subbandCount,
                                                 width, height, readFrames);

            decodeAndDequantize(subbandData, subbandCount, quantizerType, arithCoderModelType, arithCoder,
                                quantArithModel);
            TimersData::getInstance()->stopTimer("dequantizeDecode");

            TimersData::getInstance()->startTimer("delinearize");
            transform.delinearize3D(imageSequence->getChannelData(channelIndex), dequantizedData, width, height,
                                    readFrames, levles);
            TimersData::getInstance()->stopTimer("delinearize");

            TimersData::getInstance()->startTimer("cudaHostDevice");
            cutilSafeCall(cudaMemcpy(deviceInput, dequantizedData, width * height * readFrames * sizeof(float),
                                     cudaMemcpyHostToDevice));
            TimersData::getInstance()->stopTimer("cudaHostDevice");

            TimersData::getInstance()->startTimer("cudaTransform");
            transform.reverseDeviceMemory3D(deviceInput, deviceOutput[channelIndex], width, height, readFrames,
                                            levles, wavelet);
            TimersData::getInstance()->stopTimer("cudaTransform");
        }

        TimersData::getInstance()->startTimer("cudaColorTransform");
        if (colorTransformType == YCBCR_COLOR_TRANSFORM) {
            YCbCrColorTransformCuda colorTransform;
            colorTransform.transformReverseDevice(deviceOutput, width * height * readFrames);
        }
        TimersData::getInstance()->stopTimer("cudaColorTransform");

        TimersData::getInstance()->startTimer("cudaDeviceHost");
        for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
             cutilSafeCall(cudaMemcpy(imageSequence->getChannelData(channelIndex), deviceOutput[channelIndex],
                                     width * height * readFrames * sizeof(float), cudaMemcpyDeviceToHost));
        }
        TimersData::getInstance()->stopTimer("cudaDeviceHost");

        TimersData::getInstance()->startTimer("saveImageSequence");
        imageSequence->flushImageSequence();
        TimersData::getInstance()->stopTimer("saveImageSequence");
    }

    cudaFree(deviceInput);
    for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
        cudaFree(deviceOutput[channelIndex]);
    }

    delete wavelet;
    delete quantArithModel;
    delete [] dequantizedData;

    cudaThreadExit();
}
