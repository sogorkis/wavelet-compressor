/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef EMPTYCOLORTRANSFORM_H
#define EMPTYCOLORTRANSFORM_H

#include "ColorTransform.h"

class EmptyColorTransform : public ColorTransform {
public:
    virtual void transformForward(float ** , int ) { /* do nothing */ }

    virtual void transformReverse(float ** , int ) { /* do nothing */ }

    virtual float getChannelBudgetRatio(int ) { return 1.0f; }
};

#endif // EMPTYCOLORTRANSFORM_H
