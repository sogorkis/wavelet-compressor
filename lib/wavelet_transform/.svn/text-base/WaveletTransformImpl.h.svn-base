/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef WAVELETTRANSFORMIMPL_H
#define WAVELETTRANSFORMIMPL_H

#include "WaveletTransform.h"

/**
  * Normal CPU, convolution based wavelet transform implementation.
  */
class WaveletTransformImpl : public WaveletTransform
{
public:
    virtual void forward1D(const float * input, float * output, int length, int levels, Wavelet * wavelet);

    virtual void reverse1D(const float * input, float * output, int length, int levels, Wavelet * wavelet);

    virtual void forward2D(const float * input, float * output, int width, int height, int levels, Wavelet * wavelet);

    virtual void reverse2D(const float * input, float * output, int width, int height, int levels, Wavelet * wavelet);

    virtual void forward3D(const float * input, float * output, int width, int height, int frames, int levels, Wavelet * wavelet);

    virtual void reverse3D(const float * input, float * output, int width, int height, int frames, int levels, Wavelet * wavelet);

    virtual void linearize2D(const float * input, float * output, int width, int height, int levels);

    virtual void delinearize2D(const float * input, float * output, int width, int height, int levels);

    virtual void linearize3D(const float * input, float * output, int width, int height, int depth, int levels);

    virtual void delinearize3D(const float * input, float * output, int width, int height, int depth, int levels);

private:
    float getInputValue(const float *input, int index, int inputLength, int offset = 0,
                        bool ignoreOddIndex = false, bool ignoreEvenIndex = false, bool asymmetric = false);

    void transformForward(const float * input, float * output, int length, Wavelet * wavelet);

    void transformReverse(const float * input, float * output, int length, Wavelet * wavelet);

    void copyRowFrom2D(const float * input, float * output, int rowIndex, int length, int width);

    void copyRowTo2D(const float * input, float * output, int rowIndex, int length, int width);

    void copyColumnFrom2D(const float * input, float * output, int columnIndex, int width, int height);

    void copyColumnTo2D(const float * input, float * output, int columnIndex, int width, int height);


    void copyRowFrom3D(const float * input, float * output, int rowIndex, int depthIndex, int length, int width, int height);

    void copyRowTo3D(const float * input, float * output, int rowIndex, int depthIndex, int length, int width, int height);

    void copyColumnFrom3D(const float * input, float * output, int columnIndex, int depthIndex, int length, int width, int height);

    void copyColumnTo3D(const float * input, float * output, int columnIndex, int depthIndex, int length, int width, int height);

    void copyDepthsFrom3D(const float * input, float * output, int rowIndex, int columnIndex, int length, int width, int height);

    void copyDepthsTo3D(const float * input, float * output, int rowIndex, int columnIndex, int length, int width, int height);
};

#endif // WAVELETTRANSFORMIMPL_H
