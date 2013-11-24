/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef WAVELETTRANSFORMCUDAIMPL_H
#define WAVELETTRANSFORMCUDAIMPL_H

#include "WaveletTransformImpl.h"

/**
  * CUDA implementation of wavelet transform.
  */
class WaveletTransformCudaImpl : public WaveletTransformImpl {
public:
    virtual void forward1D(const float * input, float * output, int length, int levels, Wavelet * wavelet);

    virtual void reverse1D(const float * input, float * output, int length, int levels, Wavelet * wavelet);

    virtual void forward2D(const float * input, float * output, int width, int height, int levels, Wavelet * wavelet);

    virtual void reverse2D(const float * input, float * output, int width, int height, int levels, Wavelet * wavelet);

    virtual void forward3D(const float * input, float * output, int width, int height, int frames, int levels, Wavelet * wavelet);

    virtual void reverse3D(const float * input, float * output, int width, int height, int frames, int levels, Wavelet * wavelet);

    void forwardDeviceMemory1D(float *deviceInput, float *deviceOutput, int length, int levels, Wavelet * wavelet);

    void reverseDeviceMemory1D(float *deviceInput, float *deviceOutput, int length, int levels, Wavelet * wavelet);

    void forwardDeviceMemory2D(float *deviceInput, float *deviceOutput, int width, int height, int levels, Wavelet * wavelet);

    void reverseDeviceMemory2D(float *deviceInput, float *deviceOutput, int width, int height, int levels, Wavelet * wavelet);

    void forwardDeviceMemory3D(float *deviceInput, float *deviceOutput, int width, int height, int frames, int levels, Wavelet * wavelet);

    void reverseDeviceMemory3D(float *deviceInput, float *deviceOutput, int width, int height, int frames, int levels, Wavelet * wavelet);
};

#endif // WAVELETTRANSFORMCUDAIMPL_H
