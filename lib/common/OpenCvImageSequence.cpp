/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "OpenCvImageSequence.h"
#include "OpenCvImage.h"
#include "util/WaveletCompressorUtil.h"

#include <sstream>

OpenCvImageSequence::OpenCvImageSequence() {
    capture = NULL;
    writer = NULL;
    channelData = NULL;
    frameH = frameW = fps = numFrames = readCount = 0;
}

OpenCvImageSequence::~OpenCvImageSequence() {
    if(channelData != NULL) {
        for(int i = 0; i < channelCount; ++i) {
            delete [] channelData[i];
        }
        delete [] channelData;
    }
}

void OpenCvImageSequence::initReadSequence(std::string &inputFileName) {
    capture = cvCaptureFromAVI(inputFileName.c_str());

    frameH    = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
    frameW    = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
    fps       = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
    numFrames = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT);
    channelCount = 3; //FIXME
}

void OpenCvImageSequence::setWriteSequenceFileName(std::string &outputFileName) {
    this->outFileName = outputFileName;
}

void OpenCvImageSequence::initWriteSequence(int fps,
                                            int width,
                                            int height,
                                            int frames,
                                            int frameGroupSize,
                                            int channelCount) {
    this->frameH        = height;
    this->frameW        = width;
    this->numFrames     = frames;
    this->fps           = fps;
    this->channelCount  = channelCount;
    this->frameGroupSize= frameGroupSize;

    writer = cvCreateVideoWriter(outFileName.c_str(), CV_FOURCC('H','F','Y','U'), fps,
                                 cvSize(frameW, frameH), 1);

    channelData = new float*[channelCount];
    for(int i = 0; i < channelCount; ++i) {
        channelData[i] = new float[width * height * frameGroupSize];
    }
}

void OpenCvImageSequence::close() {
    if(capture != NULL) {
        cvReleaseCapture(&capture);
    }
    if(writer != NULL) {
        cvReleaseVideoWriter(&writer);
    }
}

Image * OpenCvImageSequence::getNextFrame() {
    if(!hasNextFrame()) {
        return NULL;
    }

    cvGrabFrame(capture);
    IplImage *img = cvRetrieveFrame(capture);

    OpenCvImage * image = new OpenCvImage(img);

    cvReleaseImage(&img);
    ++readCount;

    return image;
}

int OpenCvImageSequence::readImages(float **dst, int imageCount) {
    int r = 0;
    for(; r < imageCount; ++r, ++readCount) {

        if(!hasNextFrame()) {
            return r;
        }

        cvGrabFrame(capture);
        IplImage *img = cvRetrieveFrame(capture);

        int channelCount    = img->nChannels;
        int widthStep       = img->widthStep;
        for(int i = 0; i < channelCount; ++i) {
            for(int x = 0; x < frameW; ++x) {
                for(int y = 0; y < frameH; ++y) {
                    unsigned char val = img->imageData[y*widthStep + x*channelCount + i];
                    dst[i][r*frameW*frameH + y*frameW + x] = val;
                }
            }
        }

    }
    return r;
}

void OpenCvImageSequence::flushImageSequence() {
    int wFrames = 0;

    while(readCount < numFrames && wFrames < frameGroupSize) {
        IplImage *img = cvCreateImage(cvSize(frameW, frameH), IPL_DEPTH_8U, channelCount);

        for(int i = 0; i < channelCount; ++i) {
            for(int x = 0; x < frameW; ++x) {
                for(int y = 0; y < frameH; ++y) {
                    int value = channelData[i][wFrames*frameW*frameH + y*frameW + x];
                    value = value < 0 ? 0 : value;
                    value = value > 255 ? 255 : value;
                    img->imageData[y*img->widthStep + x*channelCount + i] = value;
                }
            }
        }

        cvWriteFrame(writer, img);
        cvReleaseImage(&img);

        ++readCount;
        ++wFrames;
    }
}
