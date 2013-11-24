/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef DEADZONEUNIFORMQUANTIZER_H
#define DEADZONEUNIFORMQUANTIZER_H

#include "Quantizer.h"

class DeadzoneUniformQuantizer : public Quantizer {
public:
    DeadzoneUniformQuantizer(bool removeMean) : Quantizer(removeMean) {}

    DeadzoneUniformQuantizer(bool removeMean, float min, float max, float mean, int bits = 0)
        : Quantizer(removeMean, min, max, mean, bits) {}

    virtual ~DeadzoneUniformQuantizer() {}

    virtual void quantize(const float * input, unsigned int * output, int length);

    virtual void dequantize(const unsigned int * input, float * output, int length);

    virtual void writeData(ArithCoderModel &model);

    virtual void readData(ArithCoderModel &model);

    virtual void quantize(DataIterator &iter, ArithCoderModel &model);

    virtual void dequantize(DataIterator &iter, ArithCoderModel &model);

    virtual void getRateDistortion(DataIterator &iter,
                                   ArithCoderModel &model,
                                   int bits,
                                   float &rate,
                                   float& distortion);
private:
    static const float KFACTOR = 0.5;

    void quantize(float input, unsigned int &output, float delta, float kDelta);

    void dequantize(unsigned int input, float &output, float delta);

    float getDelta();
};

#endif // DEADZONEUNIFORMQUANTIZER_H
