/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "YCbCrColorTransform.h"

void YCbCrColorTransform::transformForward(float **input, int length) {
    for(int i = 0; i < length; ++i) {
        float r = input[2][i];
        float g = input[1][i];
        float b = input[0][i];

        input[0][i] = 0   + 0.2989 * r + 0.5866 * g + 0.1145 * b; // Y
        input[1][i] = 128 - 0.1687 * r - 0.3312 * g + 0.5000 * b; // Cb
        input[2][i] = 128 + 0.5000 * r - 0.4183 * g - 0.0816 * b; // Cr
    }
}

void YCbCrColorTransform::transformReverse(float **input, int length) {
    for(int i = 0; i < length; ++i) {
        float y  = input[0][i];
        float cb = input[1][i] - 128;
        float cr = input[2][i] - 128;

        input[0][i] = y + 1.7650 * cb;                  // B
        input[1][i] = y - 0.3456 * cb - 0.7145 * cr;    // G
        input[2][i] = y + 1.4022 * cr;                  // R
    }
}

float YCbCrColorTransform::getChannelBudgetRatio(int channel) {
    return channel == 0 ? 0.8f : 0.1f;
}
