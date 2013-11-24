/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef QUANTIZER_H
#define QUANTIZER_H

#include <arithcoder/ArithCoderModel.h>
#include <common/DataIterator.h>

class Quantizer {
public:
    Quantizer(bool removeMean) : removeMean(removeMean), mean(0) {}

    Quantizer(bool removeMean, float min, float max, float mean, int bits)
        : removeMean(removeMean), min(min), max(max), mean(mean), bits(bits) {}

    virtual ~Quantizer() {}

    virtual void quantize(const float * input, unsigned int * output, int length) = 0;

    virtual void dequantize(const unsigned int * input, float * output, int length) = 0;

    virtual void writeData(ArithCoderModel &model) = 0;

    virtual void readData(ArithCoderModel &model) = 0;

    virtual void quantize(DataIterator &iter, ArithCoderModel &model) = 0;

    virtual void dequantize(DataIterator &iter, ArithCoderModel &model) = 0;

    virtual void getRateDistortion(DataIterator &iter,
                                   ArithCoderModel &model,
                                   int bits,
                                   float &rate,
                                   float& distortion) = 0;

    float getMin() { return min; }

    float getMax() { return max; }

    float getMean() { return mean; }

    int getBits() { return bits; }

protected:

    float readElementUsingBits(unsigned int input, int bits) {
        float element = 0.0f;
        if((input & (1 << (bits- 1))) != 0) {
            element = input ^ (1 << (bits - 1));
            element = -element;
        }
        else {
            element = input;
        }
        return element;
    }

    unsigned int setElementSignBit(unsigned int input, int bits, int signInput) {
        if(signInput < 0) {
            input |= (1 << (bits-1));
        }
        return input;
    }

    void fillZeros(float * output, int length) {
        for(int i = 0; i < length; ++i) {
            output[i] = 0;
        }
    }

    bool removeMean;
    float min, max, mean;
    int bits;
};

#endif // QUANTIZER_H
