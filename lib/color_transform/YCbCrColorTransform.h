/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef YCBCRCOLORTRANSFORM_H
#define YCBCRCOLORTRANSFORM_H

#include "ColorTransform.h"

class YCbCrColorTransform : public ColorTransform {
public:
    virtual void transformForward(float ** input, int length);

    virtual void transformReverse(float ** input, int length);

    virtual float getChannelBudgetRatio(int channel);
};

#endif // YCBCRCOLORTRANSFORM_H
