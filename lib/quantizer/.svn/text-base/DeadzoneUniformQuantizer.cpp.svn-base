/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "DeadzoneUniformQuantizer.h"

#include <cfloat>

inline void DeadzoneUniformQuantizer::quantize(float input, unsigned int &output, float delta, float kDelta) {
    if(removeMean) {
        input -= mean;
    }

    float absElement = fabsf(input);

    if(absElement < kDelta) {
        output = 0;
    }
    else {
        unsigned int value = absElement / delta + KFACTOR;
        output = setElementSignBit(value, bits, input);
    }
}

inline void DeadzoneUniformQuantizer::dequantize(unsigned int input, float &output, float delta) {
    float element = readElementUsingBits(input, bits);
    float absElement = fabsf(element);

    if(element == 0) {
        output = 0;
    }
    else {
        float value = (absElement) * delta;
        value *= element < 0 ? -1 : 1;
        output =  value;
    }

    if(removeMean) {
        output += mean;
    }
}


void DeadzoneUniformQuantizer::quantize(const float *input, unsigned int *output, int length) {
    float delta = getDelta();
    float kDelta = fabsf(delta * KFACTOR);

    for(int i = 0; i < length; ++i) {
        quantize(input[i], output[i], delta, kDelta);
    }
}

void DeadzoneUniformQuantizer::dequantize(const unsigned int *input, float *output, int length) {
    if(bits == 0) {
        fillZeros(output, length);
        return;
    }

    float delta = getDelta();

    for(int i = 0; i < length; ++i) {
        dequantize(input[i], output[i], delta);
    }
}

void DeadzoneUniformQuantizer::quantize(DataIterator &iter, ArithCoderModel &model) {
    if(bits == 0) {
        return;
    }

    float delta = getDelta();
    float kDelta = fabsf(delta * KFACTOR);

    while(iter.hasNext()) {
        float inputValue = iter.next();
        unsigned int quantizedValue = 0.0;
        quantize(inputValue, quantizedValue, delta, kDelta);
        model.EncodeSymbol(quantizedValue);
    }
}

void DeadzoneUniformQuantizer::dequantize(DataIterator &iter, ArithCoderModel &model) {
    if(bits == 0) {
        while(iter.hasNext()) {
            iter.setNext(0.0f);
        }
        return;
    }

    float delta = getDelta();

    while(iter.hasNext()) {
        int symbol = model.DecodeSymbol();
        float dequantized = 0.0f;
        dequantize(symbol, dequantized, delta);
        iter.setNext(dequantized);
    }
}

void DeadzoneUniformQuantizer::getRateDistortion(DataIterator &iter,
                                                 ArithCoderModel &model,
                                                 int bits,
                                                 float &rate,
                                                 float &distortion) {
    rate = distortion = 0;

    if(bits < 2) {
        while(iter.hasNext()) {
            float inputValue = iter.next() - min;
            distortion += inputValue * inputValue;
        }
        return;
    }
    this->bits = bits;

    float delta = getDelta();
    float kDelta = fabsf(delta * KFACTOR);

    if(delta < 0.05f) {
        rate = FLT_MAX;
        distortion = FLT_MAX;
        return;
    }

    while(iter.hasNext()) {
        float inputValue = iter.next();
        unsigned int quantizedValue = 0.0;
        float dequantized = 0.0f;
        quantize(inputValue, quantizedValue, delta, kDelta);
        dequantize(quantizedValue, dequantized, delta);

        float diff = inputValue - dequantized;
        distortion += diff * diff;

        rate += model.getEncodeCost(quantizedValue);
    }
}

void DeadzoneUniformQuantizer::writeData(ArithCoderModel &model) {
    model.encodeFloat(min);
    model.encodeFloat(max);
    model.encodeUChar(bits);
    if(removeMean) {
        model.encodeFloat(mean);
    }
}

void DeadzoneUniformQuantizer::readData(ArithCoderModel &model) {
    min = model.decodeFloat();
    max = model.decodeFloat();
    bits = model.decodeUChar();
    if(removeMean) {
        mean = model.decodeFloat();
    }
}

float DeadzoneUniformQuantizer::getDelta() {
    float absMax = fabsf(removeMean ? max - mean : max);
    float absMin = fabsf(removeMean ? min - mean : min);
    double a = absMax > absMin ? absMax : absMin;
    return a / ((1 << (bits - 1)) - 1);
}
