/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef CUDAWAVELETCOMPRESSOR_H
#define CUDAWAVELETCOMPRESSOR_H

#include "WaveletCompressor.h"

class CudaWaveletCompressor : public WaveletCompressor {
public:
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
};

#endif // CUDAWAVELETCOMPRESSOR_H
