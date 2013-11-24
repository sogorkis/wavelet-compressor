/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef ARITHCODERMODELFACTORY_H
#define ARITHCODERMODELFACTORY_H

#include "ArithCoderModelOrder0.h"

/**
  * Enum used by ArithCoderModelFactory method getInstance()
  */
enum ArithCoderModelType {
    MODEL_ORDER0
};

/**
  * Factory class constructing ArithCoderModel instances.
  */
class ArithCoderModelFactory {
public:

    /**
      * Returns ArithCoderModel instance.
      * @param mAC ArithmeticCoder instance
      * @param type type of ArithCoderModel
      * @param bits bits per symbol
      * @return ArithCoderModel instance
      */
    static ArithCoderModel * getInstance(ArithCoder *mAC, ArithCoderModelType type, int bits) {
        switch (type) {
        case MODEL_ORDER0: return new ArithCoderModelOrder0(mAC, bits);
        }
        return NULL;
    }
};

#endif // ARITHCODERMODELFACTORY_H
