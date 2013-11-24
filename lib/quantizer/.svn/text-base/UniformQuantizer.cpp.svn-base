/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "UniformQuantizer.h"
#include <cfloat>

void UniformQuantizer::quantize(const float *input, unsigned int *output, int length) {
    float quantizationInterval = getQuantizetionInterval();

    for(int i = 0; i < length; ++i) {
        float inputValue = input[i];
        int l = (inputValue - min) / quantizationInterval + 0.5;
        output[i] = l;
    }
}

void UniformQuantizer::dequantize(const unsigned int * input, float * output, int length) {
    if(bits == 0) {
        fillZeros(output, length);
        return;
    }

    float quantizationInterval = getQuantizetionInterval();

    for(int i = 0; i < length; ++i) {
        float l = input[i] * quantizationInterval + min;
        output[i] = l;
    }
}

void UniformQuantizer::quantize(DataIterator &iter, ArithCoderModel &model) {
    if(bits == 0) {
        return;
    }
    float quantizationInterval = getQuantizetionInterval();

    while(iter.hasNext()) {
        float inputValue = iter.next();
        int quantizedValue = (inputValue - min) / quantizationInterval + 0.5;
        model.EncodeSymbol(quantizedValue);
    }
}

void UniformQuantizer::dequantize(DataIterator &iter, ArithCoderModel &model) {
    if(bits == 0) {
        while(iter.hasNext()) {
            iter.setNext(0.0f);
        }
        return;
    }

    float quantizationInterval = getQuantizetionInterval();

    while(iter.hasNext()) {
        int symbol = model.DecodeSymbol();
        float dequantized = symbol * quantizationInterval + min;
        iter.setNext(dequantized);
    }
}

void UniformQuantizer::getRateDistortion(DataIterator &iter, ArithCoderModel &model,
                                         int bits, float &rate, float &distortion) {
    rate = distortion = 0;

    if(bits == 0) {
        while(iter.hasNext()) {
            float inputValue = iter.next() - min;
            distortion += inputValue * inputValue;
        }
        return;
    }

    this->bits = bits;
    float quantizationInterval = getQuantizetionInterval();

    if(quantizationInterval < 0.05f) {
        rate = FLT_MAX;
        distortion = FLT_MAX;
        return;
    }

    while(iter.hasNext()) {
        float inputValue = iter.next();
        int quantizedValue = (inputValue - min) / quantizationInterval + 0.5;
        float dequantized = quantizedValue * quantizationInterval + min;

        float diff = inputValue - dequantized;
        distortion += diff * diff;

        rate += model.getEncodeCost(quantizedValue);
    }
}

void UniformQuantizer::writeData(ArithCoderModel &model) {
    model.encodeFloat(min);
    model.encodeFloat(max);
    model.encodeUChar(bits);
    if(removeMean) {
        model.encodeFloat(mean);
    }
}

void UniformQuantizer::readData(ArithCoderModel &model) {
    min = model.decodeFloat();
    max = model.decodeFloat();
    bits = model.decodeUChar();
    if(removeMean) {
        mean = model.decodeFloat();
    }
}

float UniformQuantizer::getQuantizetionInterval() {
    float absMax = fabsf(removeMean ? max - mean : max);
    float absMin = fabsf(removeMean ? min - mean : min);
    double a = absMax > absMin ? absMax : absMin;
    return a / ((1 << (bits - 1)) - 1);
}
