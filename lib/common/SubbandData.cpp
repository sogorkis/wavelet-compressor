/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "SubbandData.h"

SubbandData::SubbandData(float *data, int subbandCount)
    : subbandCount(subbandCount) {
    startPtrs       = new float*[subbandCount];
    subbandLengths  = new int[subbandCount];
    startPtrs[0]    = data;
}

SubbandData::~SubbandData() {
    delete [] startPtrs;
    delete [] subbandLengths;
}

DataIterator * SubbandData::getDataIteratorForSubband(int subbandIndex) {
    return new DataIterator(startPtrs[subbandIndex], startPtrs[subbandIndex] + subbandLengths[subbandIndex]);
}

ImageSubbandData::ImageSubbandData(float *data, int subbandCount, int width, int height)
    : SubbandData(data, subbandCount), width(width), height(height) {

    for(int subbandIndex = 0; subbandIndex < subbandCount; ++subbandIndex) {
        int subbandWidth = getSubbandWidth2D(subbandIndex, subbandCount, width);
        int subbandHeight = getSubbandHeight2D(subbandIndex, subbandCount, height);
        subbandLengths[subbandIndex] = subbandWidth * subbandHeight;

        if(subbandIndex == 0) {
            continue;
        }

        startPtrs[subbandIndex] = startPtrs[subbandIndex - 1] + subbandLengths[subbandIndex - 1];
    }

}

ImageSequenceSubbandData::ImageSequenceSubbandData(float *data, int subbandCount, int width, int height, int frames)
    : SubbandData(data, subbandCount), width(width), height(height), frames(frames) {

    debug("-----------Subband sizes 3D------------");
    for(int subbandIndex = 0; subbandIndex < subbandCount; ++subbandIndex) {
        int subbandWidth = getSubbandWidth3D(subbandIndex, subbandCount, width);
        int subbandHeight = getSubbandHeight3D(subbandIndex, subbandCount, height);
        int subbandFrames = getSubbandFrames3D(subbandIndex, subbandCount, frames);
        subbandLengths[subbandIndex] = subbandWidth * subbandHeight * subbandFrames;

        debug("%3d | %4d | %4d | %4d | %10d |", subbandIndex, subbandWidth, subbandHeight,
              subbandFrames, subbandLengths[subbandIndex]);

        if(subbandIndex == 0) {
            continue;
        }

        startPtrs[subbandIndex] = startPtrs[subbandIndex - 1] + subbandLengths[subbandIndex - 1];
    }
    debug("---------------------------------------");
}
