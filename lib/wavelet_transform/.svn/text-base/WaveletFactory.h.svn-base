/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef WAVELETFACTORY_H
#define WAVELETFACTORY_H

#include <string>

#include "Wavelet.h"

/**
  * Class constructing wavelet instances.
  */
class WaveletFactory {
public:
    /**
      * Returns wavelet instance for specified wavelet name. Returns NULL if
      * wavelet name is not valid.
      * @param waveletName name of wavelet
      * @return wavelet instance
      */
    static Wavelet * getInstance(const std::string &waveletName);

    /**
      * Array containing wavelet names.
      */
    static const char* waveletNames[];

    /**
      * WaveletNames array length.
      */
    static const int waveletNamesLength;
private:
    static void divideSqrt2(float * input, int length);
    static void multiplySqrt2(float * input, int length);
};

#endif // WAVELETFACTORY_H
