/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef UNIFORMQUANTIZER_H
#define UNIFORMQUANTIZER_H

#include "Quantizer.h"

class UniformQuantizer : public Quantizer {
public:
    UniformQuantizer(bool removeMean) : Quantizer(removeMean) {}

    UniformQuantizer(bool removeMean, float min, float max, float mean, int bits = 0)
        : Quantizer(removeMean, min, max, mean, bits) {}

    virtual ~UniformQuantizer() {}

    virtual void quantize(const float * input, unsigned int * output, int length);

    virtual void dequantize(const unsigned int * input, float * output, int length);

    virtual void writeData(ArithCoderModel &model);

    virtual void readData(ArithCoderModel &model);

    virtual void quantize(DataIterator &iter, ArithCoderModel &model);

    virtual void dequantize(DataIterator &iter, ArithCoderModel &model);

    virtual void getRateDistortion(DataIterator &iter, ArithCoderModel &model, int bits, float &rate, float& distortion);
private:
    float getQuantizetionInterval();
};

#endif // UNIFORMQUANTIZER_H
