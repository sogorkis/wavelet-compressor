/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef OPENCVIMAGESEQUENCE_H
#define OPENCVIMAGESEQUENCE_H

#include "ImageSequence.h"

#include <opencv/highgui.h>

class OpenCvImageSequence : public ImageSequence {
public:
    OpenCvImageSequence();

    virtual ~OpenCvImageSequence();

    void initReadSequence(std::string &inputFileName);

    void setWriteSequenceFileName(std::string &outputFileName);

    void close();

    virtual void initWriteSequence(int fps, int width, int height, int frames, int frameGroupSize, int channelCount);

    virtual int getFrameCount() { return numFrames; }

    virtual Image * getNextFrame();

    virtual bool hasNextFrame() { return readCount < numFrames; }

    virtual int readImages(float ** dst, int imageCount);

    virtual int getHeight() { return frameH; }

    virtual int getWidth() { return frameW; }

    virtual int getFps() { return fps; }

    virtual void flushImageSequence();

    virtual int getChannelCount() { return channelCount; }
private:
    CvCapture *capture;
    int frameH, frameW, fps, numFrames, channelCount, frameGroupSize, readCount;

    std::string outFileName;
    CvVideoWriter *writer;

    OpenCvImageSequence(const OpenCvImageSequence& o);
    OpenCvImageSequence& operator=(const OpenCvImageSequence& o);
};

#endif // OPENCVIMAGESEQUENCE_H
