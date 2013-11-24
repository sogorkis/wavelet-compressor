/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef IMAGESEQUENCE_H
#define IMAGESEQUENCE_H

#include "Image.h"

class ImageSequence {
public:
    virtual ~ImageSequence() {}

    virtual int getFrameCount() = 0;

    virtual Image * getNextFrame() = 0;

    virtual bool hasNextFrame() = 0;

    virtual int readImages(float ** dst, int imageCount) = 0;

    virtual int getHeight() = 0;

    virtual int getWidth() = 0;

    virtual int getFps() = 0;

    virtual int getChannelCount() = 0;

    virtual void initWriteSequence(int fps, int width, int height, int frames, int frameGroupSize,  int channelCount) = 0;

    virtual void flushImageSequence() = 0;

    float ** getChannelData() { return channelData; }

    float * getChannelData(int channelIndex) { return channelData[channelIndex]; }
protected:
    float ** channelData;
};

#endif // IMAGESEQUENCE_H
