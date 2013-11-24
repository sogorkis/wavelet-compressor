/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef OPENCVIMAGE_H
#define OPENCVIMAGE_H

#include "Image.h"

#include <opencv/cv.h>

class OpenCvImage : public Image {
public:
    OpenCvImage() {}

    OpenCvImage(IplImage *cvImage);

    virtual void init(int width, int height, int channelCount);

    IplImage * getIplImage();
};

#endif // OPENCVIMAGE_H
