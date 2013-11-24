/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "WaveletFactory.h"

#include <cmath>

const char* WaveletFactory::waveletNames[] = {"haar", "daub4", "cdf97", "antonini"};
const int WaveletFactory::waveletNamesLength = 4;

Wavelet * WaveletFactory::getInstance(const std::string &waveletName)
{
    if(waveletName == "cdf97") {
        float analysisLowPass[] = {0.026748757411,
                                   -0.016864118443,
                                   -0.078223266529,
                                   0.266864118443,
                                   0.602949018236,
                                   0.266864118443,
                                   -0.078223266529,
                                   -0.016864118443,
                                   0.026748757411};
        float analysisHighPass[] = {0.091271763114,
                                    -0.057543526229,
                                    -0.591271763114,
                                    1.11508705,
                                    -0.591271763114,
                                    -0.057543526229,
                                    0.091271763114};
        float synthesisLowPass[] = {-0.091271763114,
                                    -0.057543526229,
                                    0.591271763114,
                                    1.11508705,
                                    0.591271763114,
                                    -0.057543526229,
                                    -0.091271763114};
        float synthesisHighPass[] = {0.026748757411,
                                     0.016864118443,
                                     -0.078223266529,
                                     -0.266864118443,
                                     0.602949018236,
                                     -0.266864118443,
                                     -0.078223266529,
                                     0.016864118443,
                                     0.026748757411};
        multiplySqrt2(analysisLowPass, 9);
        divideSqrt2(analysisHighPass, 7);
        divideSqrt2(synthesisLowPass, 7);
        multiplySqrt2(synthesisHighPass, 9);
        return new Wavelet("cdf97", 9, 7, 7, 9, 4, 2/*TODO: should be 3*/, 3, 4, analysisLowPass, analysisHighPass,
                           synthesisLowPass, synthesisHighPass, true);
    }
    else if(waveletName == "haar") {
        float scales[] = {0.707106781, 0.707106781};
        float coefficients[] = {0.707106781, -0.707106781};
        return new Wavelet("haar", 2, coefficients, scales);
    }
    else if(waveletName == "daub4") {
        float scales[] = {0.6830127, 1.1830127, 0.3169873, -0.1830127};
        float coefficients[] = {-0.1830127, -0.3169873, 1.1830127, -0.6830127};
        divideSqrt2(scales, 4);
        divideSqrt2(coefficients, 4);
        return new Wavelet("daub4", 4, 4, 4, 4, 0, 0, 0, 0, scales, coefficients, scales, coefficients, false);
    }
    else if(waveletName == "antonini") {
        // 7/9 Filter from M. Antonini, M. Barlaud, P. Mathieu, and
        // I. Daubechies, "Image coding using wavelet transform", IEEE
        // Transactions on Image Processing", Vol. pp. 205-220, 1992.
        float analysisLow[] = {  3.782845550699535e-02,
                                  -2.384946501937986e-02,
                                  -1.106244044184226e-01,
                                  3.774028556126536e-01,
                                  8.526986790094022e-01,
                                  3.774028556126537e-01,
                                  -1.106244044184226e-01,
                                  -2.384946501937986e-02,
                                  3.782845550699535e-02};
        float analysisHigh[] = { -6.453888262893856e-02,
                                  4.068941760955867e-02,
                                  4.180922732222124e-01,
                                  -7.884856164056651e-01,
                                  4.180922732222124e-01,
                                  4.068941760955867e-02,
                                  -6.453888262893856e-02};
        float synthesisLow[] = { -6.453888262893856e-02,
                                  -4.068941760955867e-02,
                                  4.180922732222124e-01,
                                  7.884856164056651e-01,
                                  4.180922732222124e-01,
                                  -4.068941760955867e-02,
                                  -6.453888262893856e-02};
        float synthesisHigh[] = {  -3.782845550699535e-02,
                                    -2.384946501937986e-02,
                                    1.106244044184226e-01,
                                    3.774028556126536e-01,
                                    -8.526986790094022e-01,
                                    3.774028556126537e-01,
                                    1.106244044184226e-01,
                                    -2.384946501937986e-02,
                                    -3.782845550699535e-02};
        return new Wavelet("antonini", 9, 7, 7, 9, 4, 2/*TODO: should be 3*/, 3, 4, analysisLow, analysisHigh, synthesisLow, synthesisHigh, true);
    }
    return NULL;
}

void WaveletFactory::divideSqrt2(float *input, int length) {
    float sqrt2 = sqrt(2);
    for(int i = 0; i < length; ++i) {
        input[i] /= sqrt2;
    }
}

void WaveletFactory::multiplySqrt2(float *input, int length) {
    float sqrt2 = sqrt(2);
    for(int i = 0; i < length; ++i) {
        input[i] *= sqrt2;
    }
}
