/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef WAVELET_H
#define WAVELET_H

#include "Filter.h"

#include <string>

/**
  * Class holding data about wavelet. Wavelet class instance has four filters
  * (analysis low pass, analysis high pass, synthesis low pass, synthesis high pass)
  * which are used to compute DWT and IDWT. It also contains information whether filter
  * banks are symmetric.
  */
class Wavelet {
public:

    /**
      * Constructs Wavelet class instance using provied filter data.
      * @param name wavelet name
      * @param analysisLowLength analysis low filter length
      * @param analysisHighLength analysis high filter length
      * @param synthesisLowLength synthesis low filter length
      * @param synthesisHighLength synthesis high filter length
      * @param analysisLowFirstIndex analysis low filter first index
      * @param analysisHighFirstIndex analysis high filter first index
      * @param synthesisLowFirstIndex synthesis low filter first index
      * @param synthesisHighFirstIndex synthesis high filter first index
      * @param analysisLowPass analysis low filter values
      * @param analysisHighPass analysis high filter values
      * @param synthesisLowPass synthesis low filter values
      * @param synthesisHighPass synthesis high filter values
      * @param symmetric symmetric filter banks
      */
    Wavelet(const std::string &name,
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
            bool symmetric);

    /**
      * Constructor for simple orthogonal wavelets (eg. Haar). Equivalent of explicitly calling contructor:
      * <pre>
      * init(length, length, length, length, 0, 0, 0, 0, scales, coefficients, scales, coefficients, false);
      * </pre>
      * @param name wavelet name
      * @param length length of coefficient and scales array values
      * @param coefficients wavelet coefficients values
      * @param scales wavelet scales values
      */
    Wavelet(const std::string &name,
            int length,
            const float * coefficients,
            const float * scales);

    virtual ~Wavelet();

    /**
      * Returns wavelet analysis low pass filter.
      * @return analysis low pass filter
      */
    const Filter * getAnalysisLowFilter() { return analysisLowFilter; }

    /**
      * Returns wavelet analysis high pass filter.
      * @return analysis high pass filter
      */
    const Filter * getAnalysisHighFilter() { return analysisHighFilter; }

    /**
      * Returns wavelet synthesis low pass filter.
      * @return synthesis low pass filter
      */
    const Filter * getSynthesisLowFilter() { return synthesisLowFilter; }

    /**
      * Returns wavelet synthesis low pass filter.
      * @return synthesis low pass filter
      */
    const Filter * getSynthesisHighFilter() { return synthesisHighFilter; }

    /**
      * Returns whether wavelet filter banks are symmetric.
      * @return true if filter banks are symmetric
      */
    bool isSymmetric() { return symmetric; }

    const std::string & getName() { return *name; }
private:
    void init(const std::string &name,
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
              bool symmetric);

    std::string * name;
    bool symmetric;
    Filter * analysisLowFilter, * analysisHighFilter, * synthesisLowFilter, * synthesisHighFilter;
};

#endif // WAVELET_H
