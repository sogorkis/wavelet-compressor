/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "OpenCvImage.h"
#include "util/WaveletCompressorUtil.h"

#include <limits>

OpenCvImage::OpenCvImage(IplImage *cvImage) {
    init(cvImage->width, cvImage->height, cvImage->nChannels);

    for(int i = 0; i < channelCount; ++i) {
        for(int x = 0; x < width; ++x) {
            for(int y = 0; y < height; ++y) {
                unsigned char val = cvImage->imageData[y*cvImage->widthStep + x*channelCount + i];
                channelData[i][y * width + x] = val;
            }
        }
    }
}

void OpenCvImage::init(int width, int height, int channelCount) {
    this->channelCount = channelCount;
    this->width = width;
    this->height = height;
    this->channelData = new float*[channelCount];
    for(int i = 0; i < channelCount; ++i) {
        channelData[i] = new float[height * width];
    }
}

IplImage * OpenCvImage::getIplImage() {
    IplImage *img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, channelCount);

    for(int i = 0; i < channelCount; ++i) {
        for(int x = 0; x < width; ++x) {
            for(int y = 0; y < height; ++y) {
                int value = channelData[i][y * width + x];
                value = value < 0 ? 0 : value;
                value = value > 255 ? 255 : value;
                img->imageData[y*img->widthStep + x*channelCount + i] = value;
            }
        }
    }
    return img;
}
