/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef WAVELETTRANSFORM_H
#define WAVELETTRANSFORM_H

#include "Wavelet.h"

/**
  * Abstract class describing interface for calculation of discrete wavelet transform (DWT) and
  * inverse discrete wavelet transform (IDWT). It is assumed that output length is equal to input
  * length after DWT computation and that DWT coefficients allows precise reconstruction on signal
  * boundaries. To accomplish this requirements implementations should use periodic or symmetric
  * signal extension.<br/><br/>
  * (more information about signal extension can be found in "Classification of nonexpansive symmetric
  * extension transforms for multirate filter banks," Chris Brislawn).
  */
class WaveletTransform {
public:
    /**
      * Apply forward one dimensional transform using specified levels of decomposition and provided
      * wavelet instance.
      * @param input input values
      * @param output wavelet coefficients (aproximation cooeficients first [L L L H H H])
      * @param length input lenght
      * @param levels number of levels of decomposition
      * @param wavelet wavelet used in transform
      */
    virtual void forward1D(const float * input, float * output, int length, int levels, Wavelet * wavelet) = 0;

    /**
      * Apply reverse one dimensional transform using specified levels of decomposition and provided
      * wavelet instance.
      * @param input input values (aproximation cooeficients first [L L L H H H])
      * @param output wavelet coefficients
      * @param length input lenght
      * @param levels number of levels of decomposition
      * @param wavelet wavelet used in transform
      */
    virtual void reverse1D(const float * input, float * output, int length, int levels, Wavelet * wavelet) = 0;

    /**
      * Apply forward two dimensional transform using specified levels of decomposition and provided
      * wavelet instance.
      * @param input input values row by row
      * @param output wavelet coefficients
      * <pre>
      * |LL LL LH LH|
      * |LL LL LH LH|
      * |LH LH HH HH|
      * |LH LH HH HH|
      * </pre>
      * @param width number of columns
      * @param height number of rows
      * @param levels number of levels of decomposition
      * @param wavelet wavelet used in transform
      */
    virtual void forward2D(const float * input, float * output, int width, int height, int levels, Wavelet * wavelet) = 0;

    /**
      * Apply reverse two dimensional transform using specified levels of decomposition and provided
      * wavelet instance.
      * @param input input values
      * <pre>
      * |LL LL LH LH|
      * |LL LL LH LH|
      * |LH LH HH HH|
      * |LH LH HH HH|
      * </pre>
      * @param output wavelet coefficients
      * @param width number of columns
      * @param height number of rows
      * @param levels number of levels of decomposition
      * @param wavelet wavelet used in transform
      */
    virtual void reverse2D(const float * input, float * output, int width, int height, int levels, Wavelet * wavelet) = 0;

    /**
      * Apply forward three dimensional transform using specified levels of decomposition and provided
      * wavelet instance.
      * @param input input values
      * @param output wavelet coefficients
      * @param width number of columns
      * @param height number of rows
      * @param frames number of frames
      * @param levels number of levels of decomposition
      * @param wavelet wavelet used in transform
      */
    virtual void forward3D(const float * input, float * output, int width, int height, int frames, int levels, Wavelet * wavelet) = 0;

    /**
      * Apply reverse three dimensional transform using specified levels of decomposition and provided
      * wavelet instance.
      * @param input input values
      * @param output wavelet coefficients
      * @param width number of columns
      * @param height number of rows
      * @param frames number of frames
      * @param levels number of levels of decomposition
      * @param wavelet wavelet used in transform
      */
    virtual void reverse3D(const float * input, float * output, int width, int height, int frames, int levels, Wavelet * wavelet) = 0;

    virtual void linearize2D(const float * input, float * output, int width, int height, int levels) = 0;

    virtual void delinearize2D(const float * input, float * output, int width, int height, int levels) = 0;

    virtual void linearize3D(const float * input, float * output, int width, int height, int depth, int levels) = 0;

    virtual void delinearize3D(const float * input, float * output, int width, int height, int depth, int levels) = 0;
};

#endif // WAVELETTRANSFORM_H
