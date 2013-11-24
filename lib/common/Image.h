/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef IMAGE_H
#define IMAGE_H

class Image {
public:
    virtual void init(int width, int height, int channelCount) = 0;

    int getWidth() { return width; }

    int getHeight() { return height; }

    int getChannelCount() { return channelCount; }

    float ** getChannelData() { return channelData; }

    float * getChannelData(int channelIndex) { return channelData[channelIndex]; }

protected:
    float ** channelData;
    int width, height, channelCount;
};

#endif // IMAGE_H
