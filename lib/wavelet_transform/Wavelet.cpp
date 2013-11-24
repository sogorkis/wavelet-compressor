/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "Wavelet.h"

Wavelet::Wavelet(const std::string &name,
                 int analysisLowLength,
                 int analysisHighLength,
                 int synthesisLowLength,
                 int synthesisHighLength,
                 int analysisLowFirstIndex,
                 int analysisHighFirstIndex,
                 int synthesisLowFirstIndex,
                 int synthesisHighFirstIndex,
                 const float * analysisLowPass,
                 const float * analysisHighPass,
                 const float * synthesisLowPass,
                 const float * synthesisHighPass,
                 bool symmetric) {
    init(name, analysisLowLength, analysisHighLength, synthesisLowLength, synthesisHighLength,
         analysisLowFirstIndex, analysisHighFirstIndex, synthesisLowFirstIndex, synthesisHighFirstIndex,
         analysisLowPass, analysisHighPass, synthesisLowPass, synthesisHighPass, symmetric);
}

Wavelet::Wavelet(const std::string &name, int length, const float *coefficients, const float *scales) {
    init(name, length, length, length, length, 0, 0, 0, 0, scales, coefficients, scales, coefficients, false);
}

Wavelet::~Wavelet() {
    delete analysisLowFilter;
    delete analysisHighFilter;
    delete synthesisLowFilter;
    delete synthesisHighFilter;
    delete name;
}

void Wavelet::init(const std::string &name,
                   int analysisLowLength,
                   int analysisHighLength,
                   int synthesisLowLength,
                   int synthesisHighLength,
                   int analysisLowFirstIndex,
                   int analysisHighFirstIndex,
                   int synthesisLowFirstIndex,
                   int synthesisHighFirstIndex,
                   const float * analysisLowPass,
                   const float * analysisHighPass,
                   const float * synthesisLowPass,
                   const float * synthesisHighPass,
                   bool symmetric) {
    this->name = new std::string(name);
    this->symmetric = symmetric;
    this->analysisLowFilter = new Filter(analysisLowPass, analysisLowLength, analysisLowFirstIndex);
    this->analysisHighFilter = new Filter(analysisHighPass, analysisHighLength, analysisHighFirstIndex);
    this->synthesisLowFilter = new Filter(synthesisLowPass, synthesisLowLength, synthesisLowFirstIndex);
    this->synthesisHighFilter = new Filter(synthesisHighPass, synthesisHighLength, synthesisHighFirstIndex);
}
