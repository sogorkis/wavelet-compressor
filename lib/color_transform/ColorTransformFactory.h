/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef COLORTRANSFORMFACTORY_H
#define COLORTRANSFORMFACTORY_H

#include "YCbCrColorTransform.h"
#include "EmptyColorTransform.h"

enum ColorTransformType {
    EMPTY_COLOR_TRANSFORM,
    YCBCR_COLOR_TRANSFORM
};

class ColorTransformFactory {
public:
    static ColorTransform * getInstance(ColorTransformType type) {
        switch (type) {
        case EMPTY_COLOR_TRANSFORM:
            return new EmptyColorTransform();
        case YCBCR_COLOR_TRANSFORM:
            return new YCbCrColorTransform();
        }
    }
};

#endif // COLORTRANSFORMFACTORY_H
