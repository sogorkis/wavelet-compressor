/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef WAVELETCOMPRESSORUTIL_H
#define WAVELETCOMPRESSORUTIL_H

#include <cmath>

//#define DEBUG

void info(const char *format, ...);

void fail(const char *format, ...);

void debug(const char * format, ...);

void setVerbose(bool value);

inline int getSubbandCount1D(int decompositionLevelCount) {
    return decompositionLevelCount + 1;
}

inline int getSubbandCount2D(int decompositionLevelCount) {
    return decompositionLevelCount * 3 + 1;
}

inline int getSubbandCount3D(int decompositionLevelCount) {
    return decompositionLevelCount * 7 + 1;
}

inline int getSubbandWidth2D(int subbandIndex, int subbandCount, int imageWidth) {
    int levels = (subbandCount - 1) / 3;
    int k = (levels - (subbandIndex / 4));
    if(subbandIndex > 3) {
        k = (levels - (subbandIndex - 1) / 3);
    }
    return imageWidth / pow(2, k);
}

inline int getSubbandHeight2D(int subbandIndex, int subbandCount, int imageHeight) {
    int levels = (subbandCount - 1) / 3;
    int k = (levels - (subbandIndex / 4));
    if(subbandIndex > 3) {
        k = (levels - (subbandIndex - 1) / 3);
    }
    return imageHeight / pow(2, k);
}

inline int getSubbandXOffset2D(int subbandIndex, int subbandCount, int imageWidth) {
    if(subbandIndex == 0) {
        return 0;
    }

    int k = subbandIndex - 3 * (subbandIndex / 4);
    if(subbandIndex > 3) {
        k = subbandIndex - 3 * ((subbandIndex - 1) / 3);
    }
    if(k == 1 || k == 3) {
        return getSubbandWidth2D(subbandIndex, subbandCount, imageWidth);
    }
    return 0;
}

inline int getSubbandYOffset2D(int subbandIndex, int subbandCount, int imageHeight) {
    if(subbandIndex == 0) {
        return 0;
    }

    int k = subbandIndex - 3 * (subbandIndex / 4);
    if(subbandIndex > 3) {
        k = subbandIndex - 3 * ((subbandIndex - 1) / 3);
    }
    if(k == 2 || k == 3) {
        return getSubbandHeight2D(subbandIndex, subbandCount, imageHeight);
    }
    return 0;
}

inline int getSubbandWidth3D(int subbandIndex, int subbandCount, int width) {
    int levels = (subbandCount - 1) / 7;
    int k = (levels - (subbandIndex / 8));
    if(subbandIndex > 7) {
        k = (levels - (subbandIndex - 1) / 7);
    }
    return width / pow(2, k);
}

inline int getSubbandHeight3D(int subbandIndex, int subbandCount, int height) {
    int levels = (subbandCount - 1) / 7;
    int k = (levels - (subbandIndex / 8));
    if(subbandIndex > 7) {
        k = (levels - (subbandIndex - 1) / 7);
    }
    return height / pow(2, k);
}

inline int getSubbandFrames3D(int subbandIndex, int subbandCount, int frames) {
    int levels = (subbandCount - 1) / 7;
    int k = (levels - (subbandIndex / 8));
    if(subbandIndex > 7) {
        k = (levels - (subbandIndex - 1) / 7);
    }
    return frames / pow(2, k);
}

inline int getSubbandXOffset3D(int subbandIndex, int subbandCount, int width) {
    if(subbandIndex == 0) {
        return 0;
    }

    int k = subbandIndex - 7 * (subbandIndex / 8);
    if(subbandIndex > 7) {
        k = subbandIndex - 7 * ((subbandIndex - 1) / 8);
    }
    if(k == 1 || k == 3 || k == 5 || k == 7 || k == 8 || k == 10 || k == 12) {
        return getSubbandWidth3D(subbandIndex, subbandCount, width);
    }
    return 0;
}

inline int getSubbandYOffset3D(int subbandIndex, int subbandCount, int height) {
    if(subbandIndex == 0) {
        return 0;
    }

    int k = subbandIndex - 7 * (subbandIndex / 8);
    if(subbandIndex > 7) {
        k = subbandIndex - 7 * ((subbandIndex - 1) / 8);
    }
    if(k == 2 || k == 3 || k == 6 || k == 7 || k >= 9) {
        return getSubbandWidth3D(subbandIndex, subbandCount, height);
    }
    return 0;
}

inline int getSubbandZOffset3D(int subbandIndex, int subbandCount, int frames) {
    if(subbandIndex == 0) {
        return 0;
    }

    int k = subbandIndex - 7 * (subbandIndex / 8);
    if(subbandIndex > 7) {
        k = subbandIndex - 7 * ((subbandIndex - 1) / 8);
    }
    if(k >= 4 && k < 8) {
        return getSubbandFrames3D(subbandIndex, subbandCount, frames);
    }
    return 0;
}

#endif // WAVELETCOMPRESSORUTIL_H
