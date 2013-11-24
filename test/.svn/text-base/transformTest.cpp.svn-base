#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <cutil_inline.h>

#include "wavelet_transform/WaveletTransformImpl.h"
#include "wavelet_transform/WaveletTransformCudaImpl.h"
#include "wavelet_transform/WaveletFactory.h"
#include "util/WaveletCompressorUtil.h"

//#define DEBUG_TEST

class TransformTest : public ::testing::Test {
public:
    static bool checkArraysEqual(const float * expected, const float * actual, int length, float epsylon) {
        for(int i = 0; i < length; ++i) {
            float diff = fabs(actual[i] - expected[i]);
            if(diff > epsylon) {
                std::cout << "Index: " << i << ", expected: " << expected[i] << ", actual: " << actual[i]
                        << ", differnce: " << diff << std::endl;
                return false;
            }
        }
        return true;
    }

    void SetUp() {
        transform = new WaveletTransformImpl();
        cudaTransform = new WaveletTransformCudaImpl();
        cudaSetDevice(cutGetMaxGflopsDeviceId());
        srand(1000);
        setVerbose(true);
    }

    void TearDown() {
        cudaThreadExit();
        delete transform;
        delete cudaTransform;
    }

    static bool testWaveletInputRandom(WaveletTransform * transform, const char * waveletName, int levels, int length, int columns = 1, int frames = 1) {
        float * input = new float[length * columns * frames];
        for(int i = 0; i < length * columns * frames; ++i) {
            input[i] = rand() % 256;
        }
        bool result = testInput(transform, waveletName, levels, length, input, columns, frames);
        delete input;
        return result;
    }

    static bool testWaveletInputOrdered(WaveletTransform * transform, const char * waveletName, int levels, int length, int columns = 1, int frames = 1) {
        float * input = new float[length * columns * frames];
        for(int i = 0; i < length * columns * frames; ++i) {
            input[i] = i + 1;
        }
        bool result = testInput(transform, waveletName, levels, length, input, columns, frames);
        delete input;
        return result;
    }

protected:
    static bool testInput(WaveletTransform  * transform, const char * waveletName, int levels, int length, float * input, int columns = 1, int frames = 1) {
         Wavelet * wavelet = WaveletFactory::getInstance(waveletName);
         float * output = new float[length * columns * frames];
         float * reversed = new float[length * columns * frames];
         float epsylon = 0.25f;

         if(frames != 1) {
             transform->forward3D(input, output, columns, length, frames, levels, wavelet);
         }
         else if(columns != 1) {
            transform->forward2D(input, output, columns, length, levels, wavelet);
         } else {
            transform->forward1D(input, output, length, levels, wavelet);
        }

 #ifdef DEBUG_TEST
         for(int i = 0; i < 64 && i < length * columns * frames; ++i) {
             std:: cout << i << ": " << output[i] <<  std::endl;
         }
         std::cout << std::endl;
 #endif

         if(frames != 1) {
             transform->reverse3D(output, reversed, columns, length, frames, levels, wavelet);
         }
         else if(columns != 1) {
            transform->reverse2D(output, reversed, columns, length, levels, wavelet);
         } else {
            transform->reverse1D(output, reversed, length, levels, wavelet);
        }

 #ifdef DEBUG_TEST
         for(int i = 0; i < 64 && i < length * columns * frames; ++i) {
             std:: cout << i << ": " << input[i] << "-" << reversed[i] << std::endl;
         }
         std::cout << std::endl;
 #endif

         bool correct = checkArraysEqual(input, reversed, length * columns * frames, epsylon);

         delete wavelet;
         delete output;
         delete reversed;
         return correct;
    }

    WaveletTransform * transform;
    WaveletTransformCudaImpl * cudaTransform;
};

TEST_F(TransformTest, TestHaarTransform1D) {
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 1, 16));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 2, 16));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 3, 16));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 4, 16));
}

TEST_F(TransformTest, TestHaarTransform1D_Cuda) {
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 1, 16));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 2, 16));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 3, 16));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 4, 16));
}

TEST_F(TransformTest, TestDaub4Transform1D) {
    ASSERT_TRUE(testWaveletInputOrdered(transform, "daub4", 1, 16));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "daub4", 2, 16));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "daub4", 3, 16));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "daub4", 4, 16));
}

TEST_F(TransformTest, TestDaub4Transform1D_Cuda) {
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "daub4", 1, 16));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "daub4", 2, 16));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "daub4", 3, 16));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "daub4", 4, 16));
}

TEST_F(TransformTest, TestCdf97Transform1D) {
    ASSERT_TRUE(testWaveletInputOrdered(transform, "cdf97", 1, 16));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "cdf97", 2, 16));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "cdf97", 3, 16));
    ASSERT_TRUE(testWaveletInputRandom(transform, "cdf97", 1, 16));
    ASSERT_TRUE(testWaveletInputRandom(transform, "cdf97", 2, 16));
    ASSERT_TRUE(testWaveletInputRandom(transform, "cdf97", 3, 16));
}

TEST_F(TransformTest, TestAntoniniTransform1D) {
    ASSERT_TRUE(testWaveletInputOrdered(transform, "antonini", 1, 16));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "antonini", 2, 16));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "antonini", 3, 16));
    ASSERT_TRUE(testWaveletInputRandom(transform, "antonini", 1, 16));
    ASSERT_TRUE(testWaveletInputRandom(transform, "antonini", 2, 16));
    ASSERT_TRUE(testWaveletInputRandom(transform, "antonini", 3, 16));
}

TEST_F(TransformTest, TestAntoniniTransform1D_Cuda) {
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "antonini", 1, 16));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "antonini", 2, 16));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "antonini", 3, 16));
    ASSERT_TRUE(testWaveletInputRandom(cudaTransform, "antonini", 1, 16));
    ASSERT_TRUE(testWaveletInputRandom(cudaTransform, "antonini", 2, 16));
    ASSERT_TRUE(testWaveletInputRandom(cudaTransform, "antonini", 3, 16));
}

TEST_F(TransformTest, TestCdf97Transform1D_Cuda) {
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "cdf97", 1, 16));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "cdf97", 2, 16));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "cdf97", 3, 16));
    ASSERT_TRUE(testWaveletInputRandom(cudaTransform, "cdf97", 1, 16));
    ASSERT_TRUE(testWaveletInputRandom(cudaTransform, "cdf97", 2, 16));
    ASSERT_TRUE(testWaveletInputRandom(cudaTransform, "cdf97", 3, 16));
}

TEST_F(TransformTest, TestHaarTransform2D) {
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 1, 4, 4));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 1, 2, 6));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 1, 6, 2));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 2, 4, 4));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 2, 8, 4));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 2, 4, 8));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 3, 8, 8));
}

TEST_F(TransformTest, TestHaarTransform2D_Cuda) {
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 1, 4, 4));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 1, 2, 6));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 1, 6, 2));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 2, 4, 4));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 2, 8, 4));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 2, 4, 8));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 3, 8, 8));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 1, 512, 512));
}

TEST_F(TransformTest, TransformTestRandom16x16) {
    for(int i = 1; i < 4; ++i) {
        for(int j = 0; j < WaveletFactory::waveletNamesLength; ++j) {
            ASSERT_TRUE(testWaveletInputOrdered(transform, WaveletFactory::waveletNames[j], i, 16, 16));
        }
    }
}

TEST_F(TransformTest, TransformTestRandom16x16_Cuda) {
    for(int i = 1; i < 4; ++i) {
        for(int j = 0; j < WaveletFactory::waveletNamesLength; ++j) {
            ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, WaveletFactory::waveletNames[j], i, 16, 16));
        }
    }
}

TEST_F(TransformTest, TestHaarTransform3D) {
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 1, 2, 2, 2));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 1, 4, 8, 6));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 1, 8, 2, 4));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 2, 4, 4, 4));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 2, 16, 8, 4));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 3, 16, 16, 16));
    ASSERT_TRUE(testWaveletInputOrdered(transform, "haar", 4, 16, 16, 16));
}

TEST_F(TransformTest, TestHaarTransform3D_CUDA) {
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 1, 2, 2, 2));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 1, 4, 8, 6));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 1, 8, 2, 4));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 2, 4, 4, 4));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 2, 16, 8, 4));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 3, 16, 16, 16));
    ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, "haar", 4, 16, 16, 16));
    ASSERT_TRUE(testWaveletInputRandom(cudaTransform, "haar", 1, 128, 16, 16));
}

TEST_F(TransformTest, TestTransform3D_CUDA) {
    ASSERT_TRUE(testWaveletInputRandom(cudaTransform, "antonini", 2, 128, 256, 64));
    ASSERT_TRUE(testWaveletInputRandom(cudaTransform, "antonini", 2, 256, 128, 64));
    ASSERT_TRUE(testWaveletInputRandom(cudaTransform, "antonini", 2, 64, 128, 256));
    ASSERT_TRUE(testWaveletInputRandom(cudaTransform, "antonini", 3, 64, 256, 128));
}

TEST_F(TransformTest, TransformTestRandom16x16x16) {
    for(int i = 1; i < 4; ++i) {
        for(int j = 0; j < WaveletFactory::waveletNamesLength; ++j) {
            ASSERT_TRUE(testWaveletInputOrdered(transform, WaveletFactory::waveletNames[j], i, 16, 16, 16));
        }
    }
}

TEST_F(TransformTest, TransformTestRandom16x16x16_Cuda) {
    for(int i = 1; i < 4; ++i) {
        for(int j = 0; j < WaveletFactory::waveletNamesLength; ++j) {
            ASSERT_TRUE(testWaveletInputOrdered(cudaTransform, WaveletFactory::waveletNames[j], i, 16, 16, 16));
        }
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
