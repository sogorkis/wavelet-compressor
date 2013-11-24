/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef YCBCRCOLORTRANSFORMCUDA_H
#define YCBCRCOLORTRANSFORMCUDA_H

#include "YCbCrColorTransform.h"

class YCbCrColorTransformCuda : public YCbCrColorTransform {
public:
    virtual void transformForwardDevice(float ** input, int length);

    virtual void transformReverseDevice(float ** input, int length);
};

#endif // YCBCRCOLORTRANSFORMCUDA_H
