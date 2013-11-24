/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "WaveletCompressor.h"
#include "color_transform/ColorTransformFactory.h"
#include "wavelet_transform/WaveletTransformImpl.h"
#include "wavelet_transform/WaveletFactory.h"
#include "quantizer/UniformQuantizer.h"
#include "util/WaveletCompressorUtil.h"
#include "util/TimersData.h"
#include "arithcoder/ArithCoderModelOrder0.h"
#include "BitAllocator.h"

#include <cstdio>
#include <cassert>
#include <fstream>

const int HEADER_BYTE_COST = 20;

WaveletCompressor::WaveletCompressor() {
    totalDistortion = totalDistortionChannel[0] = totalDistortionChannel[1] = totalDistortionChannel[2]
                    = totalDistortionChannel[3] = 0;
}

void WaveletCompressor::encode(Image *image,
                               Wavelet *wavelet,
                               float targetRate,
                               std::ostream *outStream,
                               int levels,
                               QuantizerType quantizerType,
                               ArithCoderModelType arithCoderModelType,
                               ColorTransformType colorTransformType) {
    WaveletTransformImpl transform;
    ArithCoder arithCoder;
    ArithCoderModel *quantArithModel = ArithCoderModelFactory::getInstance(&arithCoder, arithCoderModelType, 8);
    ColorTransform *colorTransform = ColorTransformFactory::getInstance(colorTransformType);
    int width = image->getWidth(), height = image->getHeight();
    int channelCount    = image->getChannelCount();
    int imageSize       = height * width;
    int subbandCount    = getSubbandCount2D(levels);

    info("Encoding image - size: %dx%d, channels: %d", width, height, channelCount);
    info("wavelet: %s, decompositionLevelsCount: %d", wavelet->getName().c_str(), levels);

    float *firstBuff    = new float[imageSize];
    float *secondBuff   = new float[imageSize];

    writeImageHeader(outStream, image, wavelet, levels, quantizerType, arithCoderModelType, colorTransformType);

    arithCoder.SetOutputFile(outStream);

    TimersData::getInstance()->startTimer("colorTransform");
    colorTransform->transformForward(image->getChannelData(), width * height);
    TimersData::getInstance()->stopTimer("colorTransform");

    for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
        TimersData::getInstance()->startTimer("transform");
        transform.forward2D(image->getChannelData(channelIndex), firstBuff, width, height, levels, wavelet);
        TimersData::getInstance()->stopTimer("transform");

//        printDebugFloat(firstBuff, imageSize, width);

        TimersData::getInstance()->startTimer("linearize");
        transform.linearize2D(firstBuff, secondBuff, width, height, levels);
        TimersData::getInstance()->stopTimer("linearize");

        int byteBudget = (channelCount * imageSize * colorTransform->getChannelBudgetRatio(channelIndex)) / targetRate;
        ImageSubbandData subbandData(secondBuff, subbandCount, width, height);

        quantizeAndEncode(subbandData, subbandCount, quantizerType, arithCoderModelType,
                          arithCoder, quantArithModel, byteBudget, channelIndex);
    }

    quantArithModel->FinishEncode();

    delete [] firstBuff;
    delete [] secondBuff;
    delete colorTransform;
    delete quantArithModel;
}

void WaveletCompressor::encode(ImageSequence *imageSequence,
                               Wavelet *wavelet,
                               float targetRate,
                               std::ostream *outStream,
                               int levels,
                               QuantizerType quantizerType,
                               ArithCoderModelType arithCoderModelType,
                               ColorTransformType colorTransformType) {
    WaveletTransformImpl transform;
    ArithCoder arithCoder;
    ArithCoderModel *quantArithModel = ArithCoderModelFactory::getInstance(&arithCoder, arithCoderModelType, 8);
    ColorTransform *colorTransform = ColorTransformFactory::getInstance(colorTransformType);
    int width = imageSequence->getWidth(), height = imageSequence->getHeight();
    int channelCount    = imageSequence->getChannelCount();
    int frameCount      = imageSequence->getFrameCount();
    int frameGroupSize  = 64;
    int frameGroupCount = (frameCount - 1) / frameGroupSize + 1;
    int subbandCount    = getSubbandCount3D(levels);
    float **inputBuff   = new float*[channelCount];
    float *transfBuff   = new float[width * height * frameGroupSize];
    for(int i = 0; i < channelCount; ++i) {
        inputBuff[i] = new float[width * height * frameGroupSize];
    }

    writeImageSequenceHeader(outStream, imageSequence, wavelet, levels, quantizerType, arithCoderModelType,
                             colorTransformType);

    arithCoder.SetOutputFile(outStream);

    for(int frameGroupIndex = 0; frameGroupIndex < frameGroupCount; ++frameGroupIndex) {
        TimersData::getInstance()->startTimer("openImageSequence");
        int readFrames = imageSequence->readImages(inputBuff, frameGroupSize);
        TimersData::getInstance()->stopTimer("openImageSequence");

        TimersData::getInstance()->startTimer("colorTransform");
        colorTransform->transformForward(inputBuff, width * height * readFrames);
        TimersData::getInstance()->stopTimer("colorTransform");

        for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
            TimersData::getInstance()->startTimer("transform");
            transform.forward3D(inputBuff[channelIndex], transfBuff, width, height, readFrames, levels, wavelet);
            TimersData::getInstance()->stopTimer("transform");

            TimersData::getInstance()->startTimer("linearize");
            transform.linearize3D(transfBuff, inputBuff[channelIndex], width, height, readFrames, levels);
            TimersData::getInstance()->stopTimer("linearize");

            int byteBudget = (width * height * readFrames * channelCount
                              * colorTransform->getChannelBudgetRatio(channelIndex)) / targetRate;
            ImageSequenceSubbandData subbandData(inputBuff[channelIndex], subbandCount, width, height, readFrames);

            quantizeAndEncode(subbandData, subbandCount, quantizerType, arithCoderModelType,
                              arithCoder, quantArithModel, byteBudget, channelIndex);
        }
    }

    quantArithModel->FinishEncode();

    for(int i = 0; i < channelCount; ++i) {
        delete [] inputBuff[i];
    }
    delete [] inputBuff;
    delete [] transfBuff;
    delete colorTransform;
    delete quantArithModel;
}

void WaveletCompressor::quantizeAndEncode(SubbandData &subbandData,
                                          int subbandCount,
                                          QuantizerType quantizerType,
                                          ArithCoderModelType arithCoderModelType,
                                          ArithCoder &arithCoder,
                                          ArithCoderModel *quantArithModel,
                                          int targetBytes,
                                          int channelIndex) {
    targetBytes -= 11 * subbandCount - HEADER_BYTE_COST;
    TimersData::getInstance()->startTimer("rdOptimize");
    ArithCoderModel * arithModel;
    BitAllocator allocator;
    allocator.calculateRateDistortion(subbandData, quantizerType, arithCoderModelType);
    allocator.optimalAllocate(targetBytes, true);
    allocator.print();

    totalDistortion += allocator.getTotalDistortion();
    totalDistortionChannel[channelIndex] += allocator.getTotalDistortion();
    TimersData::getInstance()->stopTimer("rdOptimize");

    TimersData::getInstance()->startTimer("quantizeEncode");
    Quantizer **quantizers = new Quantizer*[subbandCount];

    for(int i = 0; i < subbandCount; ++i) {
        bool removeMean = i == 0;
        quantizers[i] = QuantizerFactory::getQuantizerInstance(quantizerType,
                                                               removeMean,
                                                               allocator.getSubbandMin(i),
                                                               allocator.getSubbandMax(i),
                                                               allocator.getSubbandMean(i),
                                                               allocator.getOptimalSubbandBits(i));
        quantizers[i]->writeData(*quantArithModel);
    }

    for(int subbandIndex = 0; subbandIndex < subbandCount; ++subbandIndex) {
        DataIterator *iter = subbandData.getDataIteratorForSubband(subbandIndex);

        arithModel = ArithCoderModelFactory::getInstance(&arithCoder, arithCoderModelType,
                                                         allocator.getOptimalSubbandBits(subbandIndex));

        quantizers[subbandIndex]->quantize(*iter, *arithModel);

        delete iter;
        delete arithModel;
    }

    for(int i = 0; i < subbandCount; ++i) {
        delete quantizers[i];
    }
    delete [] quantizers;
    TimersData::getInstance()->stopTimer("quantizeEncode");
}

void WaveletCompressor::decode(std::istream *inStream, Image *image) {
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
    WaveletTransformImpl transform;
    ArithCoder arithCoder;
    ArithCoderModel *quantArithModel = ArithCoderModelFactory::getInstance(&arithCoder, arithCoderModelType, 8);
    ColorTransform *colorTransform = ColorTransformFactory::getInstance(colorTransformType);
    float *firstBuff    = new float[height * width];
    float *secondBuff   = new float[height * width];

    arithCoder.SetInputFile(inStream);
    arithCoder.DecodeStart();

    for(int channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
        TimersData::getInstance()->startTimer("dequantizeDecode");
        ImageSubbandData subbandData(firstBuff, subbandCount, width, height);

        decodeAndDequantize(subbandData, subbandCount, quantizerType, arithCoderModelType, arithCoder, quantArithModel);
        TimersData::getInstance()->stopTimer("dequantizeDecode");

        TimersData::getInstance()->startTimer("delinearize");
        transform.delinearize2D(firstBuff, secondBuff, width, height, levels);
        TimersData::getInstance()->stopTimer("delinearize");

        TimersData::getInstance()->startTimer("transform");
        transform.reverse2D(secondBuff, image->getChannelData(channelIndex), width, height, levels, wavelet);
        TimersData::getInstance()->stopTimer("transform");

//        printDebugFloat(image->getChannelData(channelIndex), width * height, width);
    }

    TimersData::getInstance()->startTimer("colorTransform");
    colorTransform->transformReverse(image->getChannelData(), width * height);
    TimersData::getInstance()->stopTimer("colorTransform");

    delete wavelet;
    delete colorTransform;
    delete quantArithModel;
    delete [] firstBuff;
    delete [] secondBuff;
}

void WaveletCompressor::decode(std::istream *inStream, ImageSequence *imageSequence) {
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
    WaveletTransformImpl transform;
    ArithCoder arithCoder;
    ArithCoderModel *quantArithModel = ArithCoderModelFactory::getInstance(&arithCoder, arithCoderModelType, 8);
    ColorTransform *colorTransform = ColorTransformFactory::getInstance(colorTransformType);
    int frameGroupSize  = 64;
    int frameGroupCount = (frames - 1) / frameGroupSize + 1;
    float *dequantizedData = new float[width * height * frameGroupSize];

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

            TimersData::getInstance()->startTimer("transform");
            transform.reverse3D(dequantizedData, imageSequence->getChannelData(channelIndex), width, height,
                                readFrames, levles, wavelet);
            TimersData::getInstance()->stopTimer("transform");
        }

        TimersData::getInstance()->startTimer("colorTransform");
        colorTransform->transformReverse(imageSequence->getChannelData(), width * height * readFrames);
        TimersData::getInstance()->stopTimer("colorTransform");

        TimersData::getInstance()->startTimer("saveImageSequence");
        imageSequence->flushImageSequence();
        TimersData::getInstance()->stopTimer("saveImageSequence");
    }

    delete wavelet;
    delete colorTransform;
    delete quantArithModel;
    delete [] dequantizedData;
}

void WaveletCompressor::decodeAndDequantize(SubbandData &subbandData,
                                            int subbandCount,
                                            QuantizerType quantizerType,
                                            ArithCoderModelType arithCoderModelType,
                                            ArithCoder &arithCoder,
                                            ArithCoderModel *quantArithModel) {

    Quantizer **quantizers = new Quantizer*[subbandCount];

    info("---------------Channel statistics------------------------");
    info("| Subband |     min    |     max    |     mean   | bits |");
    for(int i = 0; i < subbandCount; ++i) {
        bool removeMean = i == 0;
        quantizers[i] = QuantizerFactory::getQuantizerInstance(quantizerType, removeMean);
        quantizers[i]->readData(*quantArithModel);
        info("|      %3d|%12.2f|%12.2f|%12.2f|  %2d  |", i, quantizers[i]->getMin(),
              quantizers[i]->getMax(), quantizers[i]->getMean(), quantizers[i]->getBits());
    }
    info("---------------------------------------------------------");

    for(int subbandIndex = 0; subbandIndex < subbandCount; ++subbandIndex) {

        ArithCoderModel *arithModel = ArithCoderModelFactory::getInstance(&arithCoder, arithCoderModelType,
                                                                          quantizers[subbandIndex]->getBits());

        DataIterator *iter = subbandData.getDataIteratorForSubband(subbandIndex);
        quantizers[subbandIndex]->dequantize(*iter, *arithModel);

        delete arithModel;
        delete iter;
    }

    for(int i = 0; i < subbandCount; ++i) {
        delete quantizers[i];
    }
    delete [] quantizers;
}

void WaveletCompressor::writeImageHeader(std::ostream *outStream,
                                         Image *image,
                                         Wavelet *wavelet,
                                         int levels,
                                         QuantizerType quantizerType,
                                         ArithCoderModelType arithCoderModelType,
                                         ColorTransformType colorTransformType) {
    writeImageHeader(outStream, image->getWidth(), image->getHeight(), image->getChannelCount(),
                     levels, wavelet, quantizerType, arithCoderModelType, colorTransformType);
}

void WaveletCompressor::writeImageHeader(std::ostream *outStream,
                                         uint16_t width,
                                         uint16_t height,
                                         uint8_t channelCount,
                                         uint8_t levels,
                                         Wavelet *wavelet,
                                         QuantizerType quantizerType,
                                         ArithCoderModelType arithCoderModelType,
                                         ColorTransformType colorTransformType) {
    uint8_t quantizerTypeId         = quantizerType;
    uint8_t arithCoderModelTypeId   = arithCoderModelType;
    uint8_t colorTransModelTypeId   = colorTransformType;
    uint8_t waveletNameLength       = wavelet->getName().length();

    outStream->write(reinterpret_cast<const char *> (&width), sizeof(uint16_t));
    outStream->write(reinterpret_cast<const char *> (&height), sizeof(uint16_t));
    outStream->write(reinterpret_cast<const char *> (&channelCount), sizeof(uint8_t));
    outStream->write(reinterpret_cast<const char *> (&levels), sizeof(uint8_t));
    outStream->write(reinterpret_cast<const char *> (&quantizerTypeId), sizeof(uint8_t));
    outStream->write(reinterpret_cast<const char *> (&arithCoderModelTypeId), sizeof(uint8_t));
    outStream->write(reinterpret_cast<const char *> (&colorTransModelTypeId), sizeof(uint8_t));
    outStream->write(reinterpret_cast<const char *> (&waveletNameLength), sizeof(uint8_t));

    const char * cstr = wavelet->getName().c_str();
    for(int i = 0; i < waveletNameLength; ++i) {
        outStream->write(reinterpret_cast<const char *> (cstr + i), sizeof(char));
    }
}

void WaveletCompressor::writeImageSequenceHeader(std::ostream *outStream,
                                                 ImageSequence *imageSequence,
                                                 Wavelet *wavelet,
                                                 int levels,
                                                 QuantizerType quantizerType,
                                                 ArithCoderModelType arithCoderModelType,
                                                 ColorTransformType colorTransformType) {
    uint32_t frames = imageSequence->getFrameCount();
    uint8_t fps = imageSequence->getFps();

    outStream->write(reinterpret_cast<const char *> (&frames), sizeof(uint32_t));
    outStream->write(reinterpret_cast<const char *> (&fps), sizeof(uint8_t));
    writeImageHeader(outStream, imageSequence->getWidth(), imageSequence->getHeight(),
                     imageSequence->getChannelCount(), levels, wavelet,
                     quantizerType, arithCoderModelType, colorTransformType);
}

void WaveletCompressor::readImageHeader(std::istream *inStream,
                                        int &width,
                                        int &height,
                                        int &channelCount,
                                        int &levels,
                                        std::string &waveletName,
                                        QuantizerType &quantizerType,
                                        ArithCoderModelType &arithCoderModelType,
                                        ColorTransformType &colorTransformType) {
    uint16_t _width, _height, _widthStep;
    uint8_t _channelCount, _decompositionLevels, _waveletNameLength;
    uint8_t _quantizerTypeId, _arithCoderModelType, _colorTransformType;

    inStream->read(reinterpret_cast<char *> (&_width), sizeof(uint16_t));
    inStream->read(reinterpret_cast<char *> (&_height), sizeof(uint16_t));
    inStream->read(reinterpret_cast<char *> (&_channelCount), sizeof(uint8_t));
    inStream->read(reinterpret_cast<char *> (&_decompositionLevels), sizeof(uint8_t));
    inStream->read(reinterpret_cast<char *> (&_quantizerTypeId), sizeof(uint8_t));
    inStream->read(reinterpret_cast<char *> (&_arithCoderModelType), sizeof(uint8_t));
    inStream->read(reinterpret_cast<char *> (&_colorTransformType), sizeof(uint8_t));
    inStream->read(reinterpret_cast<char *> (&_waveletNameLength), sizeof(uint8_t));

    for(int i = 0; i < _waveletNameLength; ++i) {
        char ch;
        inStream->read(reinterpret_cast<char *> (&ch), sizeof(char));
        waveletName.push_back(ch);
    }

    width               = _width;
    height              = _height;
    channelCount        = _channelCount;
    levels              = _decompositionLevels;
    quantizerType       = static_cast<QuantizerType>(_quantizerTypeId);
    arithCoderModelType = static_cast<ArithCoderModelType>(_arithCoderModelType);
    colorTransformType  = static_cast<ColorTransformType>(_colorTransformType);
}

void WaveletCompressor::readImageSequenceHeader(std::istream *inStream,
                                                long &frames,
                                                int &fps,
                                                int &width,
                                                int &height,
                                                int &channelCount,
                                                int &levels,
                                                std::string &waveletName,
                                                QuantizerType &quantizerType,
                                                ArithCoderModelType &arithCoderModelType,
                                                ColorTransformType &colorTransformType) {
    uint32_t _frames;
    uint8_t _fps;

    inStream->read(reinterpret_cast<char *> (&_frames), sizeof(uint32_t));
    inStream->read(reinterpret_cast<char *> (&_fps), sizeof(uint8_t));

    frames = _frames;
    fps = _fps;

    readImageHeader(inStream, width, height, channelCount, levels, waveletName,
                    quantizerType, arithCoderModelType, colorTransformType);

}

void WaveletCompressor::printDebugFloat(const float * input, int length, int lineBreak) {
    printf("    ");
    for(int i = 0; i < lineBreak; ++i) {
        printf("%8d", i);
    }
    printf("\n%2d: ", 0);
    for(int i = 0; i < length;) {
        printf("%8.2f", input[i++]);
        if(i % lineBreak == 0) {
            printf("\n%2d: ", i / lineBreak);
        }
    }
    printf("\n");
}

void WaveletCompressor::printDebugInt(const int * input, int length, int lineBreak) {
    for(int i = 0; i < length;) {
        printf("%4d", input[i++]);
        if(i % lineBreak == 0) {
            printf("\n");
        }
    }
    printf("\n");
}
