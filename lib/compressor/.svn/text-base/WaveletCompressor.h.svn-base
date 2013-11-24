/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef WAVELETCOMPRESSOR_H
#define WAVELETCOMPRESSOR_H

#include "color_transform/ColorTransformFactory.h"
#include "wavelet_transform/Wavelet.h"
#include "quantizer/QuantizerFactory.h"
#include "arithcoder/ArithCoderModelFactory.h"
#include "common/Image.h"
#include "common/ImageSequence.h"
#include "common/SubbandData.h"

#include <stdint.h>
#include <iostream>

class WaveletCompressor {
public:
    WaveletCompressor();

    virtual void encode(Image *image,
                        Wavelet *wavelet,
                        float targetRate,
                        std::ostream *outStream,
                        int levels = 5,
                        QuantizerType quantizerType = UNIFORM_QUANTIZER,
                        ArithCoderModelType arithCoderModelType = MODEL_ORDER0,
                        ColorTransformType colorTransformType = EMPTY_COLOR_TRANSFORM);

    virtual void encode(ImageSequence *imageSequence,
                        Wavelet *wavelet,
                        float targetRate,
                        std::ostream *outStream,
                        int levels = 5,
                        QuantizerType quantizerType = UNIFORM_QUANTIZER,
                        ArithCoderModelType arithCoderModelType = MODEL_ORDER0,
                        ColorTransformType colorTransformType = EMPTY_COLOR_TRANSFORM);

    virtual void decode(std::istream *inStream,
                        Image *image);

    virtual void decode(std::istream *inStream,
                        ImageSequence *imageSequence);

    float getTotalDistortion() { return totalDistortion; }

    float getTotalDistortion(int channelIndex) { return totalDistortionChannel[channelIndex]; }
protected:
    void quantizeAndEncode(SubbandData &subbandData,
                           int subbandCount,
                           QuantizerType quantizerType,
                           ArithCoderModelType arithCoderModelType,
                           ArithCoder &arithCoder,
                           ArithCoderModel *quantArithModel,
                           int targetBytes,
                           int channelIndex);

    void decodeAndDequantize(SubbandData &subbandData,
                             int subbandCount,
                             QuantizerType quantizerType,
                             ArithCoderModelType arithCoderModelType,
                             ArithCoder &arithCoder,
                             ArithCoderModel *quantArithModel);

    void readImageSequenceHeader(std::istream *inStream,
                                 long &frames,
                                 int &fps,
                                 int &width,
                                 int &height,
                                 int &channelCount,
                                 int &levels,
                                 std::string &waveletName,
                                 QuantizerType &quantizerType,
                                 ArithCoderModelType &arithCoderModelType,
                                 ColorTransformType &colorTransformType);

    void writeImageSequenceHeader(std::ostream *outStream,
                                  ImageSequence *imageSequence,
                                  Wavelet *wavelet,
                                  int levels,
                                  QuantizerType quantizerType,
                                  ArithCoderModelType arithCoderModelType,
                                  ColorTransformType colorTransformType);

    void writeImageHeader(std::ostream *outStream,
                          Image *image,
                          Wavelet *wavelet,
                          int levels,
                          QuantizerType quantizerType,
                          ArithCoderModelType arithCoderModelType,
                          ColorTransformType colorTransformType);

    void writeImageHeader(std::ostream *outStream,
                          uint16_t width,
                          uint16_t height,
                          uint8_t channelCount,
                          uint8_t levels,
                          Wavelet *wavelet,
                          QuantizerType quantizerType,
                          ArithCoderModelType arithCoderModelType,
                          ColorTransformType colorTransformType);

    void readImageHeader(std::istream *inStream,
                         int &width,
                         int &height,
                         int &channelCount,
                         int &levels,
                         std::string &waveletName,
                         QuantizerType &quantizerType,
                         ArithCoderModelType &arithCoderModelType,
                         ColorTransformType &colorTransformType);

    void printDebugFloat(const float * input, int length, int lineBreak);

    void printDebugInt(const int * input, int length, int lineBreak);

private:
    float totalDistortion, totalDistortionChannel[4];
};

#endif // WAVELETCOMPRESSOR_H
