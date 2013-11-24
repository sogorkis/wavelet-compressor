/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef SUBBANDDATAACCESOR_H
#define SUBBANDDATAACCESOR_H

#include "DataIterator.h"

class SubbandData {
public:
    SubbandData(float * data, int subbandCount);

    virtual ~SubbandData();

    virtual DataIterator * getDataIteratorForSubband(int subbandIndex);

    int getSubbandLength(int subbandIndex) { return subbandLengths[subbandIndex]; }

    int getSubbandCount() { return subbandCount; }

protected:
    float **startPtrs;
    int *subbandLengths;
    int subbandCount;
};

class ImageSubbandData : public SubbandData {
public:
    ImageSubbandData(float * data, int subbandCount, int width, int height);
private:
    int width, height;
};

class ImageSequenceSubbandData : public SubbandData {
public:
    ImageSequenceSubbandData(float * data, int subbandCount, int width, int height, int frames);
private:
    int width, height, frames;
};

#endif // SUBBANDDATAACCESOR_H
