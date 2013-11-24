/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef QUANTIZERFACTORY_H
#define QUANTIZERFACTORY_H

#include "UniformQuantizer.h"
#include "DeadzoneUniformQuantizer.h"

enum QuantizerType {
    UNIFORM_QUANTIZER,
    DEADZONE_UNIFORM_QUANTIZER
};

class QuantizerFactory {
public:
    static Quantizer* getQuantizerInstance(QuantizerType type, bool removeMean) {
        switch (type) {
        case UNIFORM_QUANTIZER:
            return new UniformQuantizer(removeMean);
        case DEADZONE_UNIFORM_QUANTIZER:
            return new DeadzoneUniformQuantizer(removeMean);
        }
        return NULL;
    }

    static Quantizer* getQuantizerInstance(QuantizerType type, bool removeMean, float min,
                                           float max, float mean, int bits=0) {
        switch (type) {
        case UNIFORM_QUANTIZER:
            return new UniformQuantizer(removeMean, min, max, mean, bits);
        case DEADZONE_UNIFORM_QUANTIZER:
            return new DeadzoneUniformQuantizer(removeMean, min, max, mean, bits);
        }
        return NULL;
    }

};

#endif // QUANTIZERFACTORY_H
