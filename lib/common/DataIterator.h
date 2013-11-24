/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef IMAGEDATAITERATOR_H
#define IMAGEDATAITERATOR_H

#include "util/WaveletCompressorUtil.h"

class DataIterator {
public:
    DataIterator(float * first, float * end) : current(first), end(end) {}

    bool hasNext() { return current != end; }

    float next() { return *current++; }

    void setNext(float value) { *current = value; ++current; }
private:
    float * current, * end;
};

#endif // IMAGEDATAITERATOR_H
