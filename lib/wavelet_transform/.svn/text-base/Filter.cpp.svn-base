/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "Filter.h"

#include <cstring>

Filter::Filter(const float * values, int length, int firstIndex) {
    this->length = length;
    this->firstIndex = firstIndex;

    this->values = new float[length];
    memcpy(this->values, values, length * sizeof(float));
}

Filter::~Filter() {
    delete [] values;
}
