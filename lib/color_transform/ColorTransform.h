/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef COLORTRANSFORM_H
#define COLORTRANSFORM_H

class ColorTransform {
public:
    virtual void transformForward(float ** input, int length) = 0;

    virtual void transformReverse(float ** input, int length) = 0;

    virtual float getChannelBudgetRatio(int channel) = 0;
};

#endif // COLORTRANSFORM_H
