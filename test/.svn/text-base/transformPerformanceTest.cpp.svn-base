#include <gtest/gtest.h>
#include <cutil_inline.h>
#include <ctime>
#include <iostream>

#include "wavelet_transform/WaveletTransformImpl.h"
#include "wavelet_transform/WaveletTransformCudaImpl.h"
#include "wavelet_transform/WaveletFactory.h"
#include "util/WaveletCompressorUtil.h"

const int PERFORMANCE_TEST_ELEMENT_NUM = 1024 * 1024;
const int PERFORMANCE_TEST_DECOMPOSITION_LEVELS = 5;

class TransformPerformanceTest : public ::testing::Test {
public:
    void SetUp() {
        normalTransformImpl = new WaveletTransformImpl();
        cudaTransformImpl   = new WaveletTransformCudaImpl();
        cudaSetDevice(cutGetMaxGflopsDeviceId());
        srand(1000);
        setVerbose(true);

        for(int i = 0; i < PERFORMANCE_TEST_ELEMENT_NUM; ++i) {
            input[i] = rand() % 255;
        }
    }

    void TearDown() {
        cudaThreadExit();
        delete normalTransformImpl;
        delete cudaTransformImpl;
    }

    void performForwardTransform1D(WaveletTransform * transform, const char * waveletName) {
        Wavelet * wavelet = WaveletFactory::getInstance(waveletName);
        transform->forward1D(input, output, PERFORMANCE_TEST_ELEMENT_NUM,
                             PERFORMANCE_TEST_DECOMPOSITION_LEVELS, wavelet);
        delete wavelet;
    }

    void performReverseTransform1D(WaveletTransform * transform, const char * waveletName) {
        Wavelet * wavelet = WaveletFactory::getInstance(waveletName);
        transform->reverse1D(output, input, PERFORMANCE_TEST_ELEMENT_NUM,
                             PERFORMANCE_TEST_DECOMPOSITION_LEVELS, wavelet);
        delete wavelet;
    }

    void performForwardTransform2D(WaveletTransform * transform, const char * waveletName) {
        Wavelet * wavelet = WaveletFactory::getInstance(waveletName);
        transform->forward2D(input, output, 1024, 1024,
                             PERFORMANCE_TEST_DECOMPOSITION_LEVELS, wavelet);
        delete wavelet;
    }

    void performReverseTransform2D(WaveletTransform * transform, const char * waveletName) {
        Wavelet * wavelet = WaveletFactory::getInstance(waveletName);
        transform->reverse2D(output, input, 1024, 1024,
                             PERFORMANCE_TEST_DECOMPOSITION_LEVELS, wavelet);
        delete wavelet;
    }

    void checkError(float *outputNormal, float *outputCuda, int elementNum) {
           float error = 0;
           for(int i = 0; i < elementNum; ++i) {
               error += fabs(outputNormal[i] - outputCuda[i]);
           }
           std::cout << "Total error: " << error << ", error per symbol: " << error / elementNum << std::endl;
    }

    WaveletTransform * normalTransformImpl;
    WaveletTransformCudaImpl * cudaTransformImpl;
    float input[PERFORMANCE_TEST_ELEMENT_NUM];
    float output[PERFORMANCE_TEST_ELEMENT_NUM];
};

TEST_F(TransformPerformanceTest, Cdf97CudaForward1D) {
    performForwardTransform1D(cudaTransformImpl, "cdf97");
}

TEST_F(TransformPerformanceTest, Cdf97CudaReverse1D) {
    performReverseTransform1D(cudaTransformImpl, "cdf97");
}

TEST_F(TransformPerformanceTest, Cdf97NormalForward1D) {
    performForwardTransform1D(normalTransformImpl, "cdf97");
}

TEST_F(TransformPerformanceTest, Cdf97NormalReverse1D) {
    performReverseTransform1D(normalTransformImpl, "cdf97");
}

TEST_F(TransformPerformanceTest, Cdf97CudaForward2D) {
    performForwardTransform2D(cudaTransformImpl, "cdf97");
}

TEST_F(TransformPerformanceTest, Cdf97CudaReverse2D) {
    performReverseTransform2D(cudaTransformImpl, "cdf97");
}

TEST_F(TransformPerformanceTest, Cdf97NormalForward2D) {
    performForwardTransform2D(normalTransformImpl, "cdf97");
}

TEST_F(TransformPerformanceTest, Cdf97NormalReverse2D) {
    performReverseTransform2D(normalTransformImpl, "cdf97");
}

TEST_F(TransformPerformanceTest, AntoniniCalculationError) {
    Wavelet * wavelet = WaveletFactory::getInstance("antonini");

    int elementNum = 1024 * 1024 * 16;
    float *input        = new float[elementNum];
    float *outputNormal = new float[elementNum];
    float *outputCuda   = new float[elementNum];
    for(int i = 0; i < elementNum; ++i) {
        input[i] = rand() % 256;
    }

    normalTransformImpl->forward1D(input, outputNormal, elementNum, 16, wavelet);
    cudaTransformImpl->forward1D(input, outputCuda, elementNum, 16, wavelet);

    std::cout << "Forward 1D" << std::endl;
    checkError(outputNormal, outputCuda, elementNum);

    memcpy(input, outputNormal, elementNum);

    normalTransformImpl->forward1D(input, outputNormal, elementNum, 16, wavelet);
    cudaTransformImpl->forward1D(input, outputCuda, elementNum, 16, wavelet);

    std::cout << "Reverse 1D" << std::endl;
    checkError(outputNormal, outputCuda, elementNum);

    for(int i = 0; i < elementNum; ++i) {
        input[i] = rand() % 256;
    }

    normalTransformImpl->forward2D(input, outputNormal, 1024, 1024, 8, wavelet);
    cudaTransformImpl->forward2D(input, outputCuda, 1024, 1024, 8, wavelet);

    std::cout << "Forward 2D" << std::endl;
    checkError(outputNormal, outputCuda, 1024 * 1024);

    memcpy(input, outputNormal, elementNum);

    normalTransformImpl->reverse2D(input, outputNormal, 1024, 1024, 8, wavelet);
    cudaTransformImpl->reverse2D(input, outputCuda, 1024, 1024, 8, wavelet);

    std::cout << "Reverse 2D" << std::endl;
    checkError(outputNormal, outputCuda, 1024 * 1024);

    for(int i = 0; i < elementNum; ++i) {
        input[i] = rand() % 256;
    }

    normalTransformImpl->forward3D(input, outputNormal, 512, 512, 64, 5, wavelet);
    cudaTransformImpl->forward3D(input, outputCuda, 512, 512, 64, 5, wavelet);

    std::cout << "Forward 3D" << std::endl;
    checkError(outputNormal, outputCuda, 512 * 512 * 64);

    memcpy(input, outputNormal, elementNum);

    normalTransformImpl->reverse3D(input, outputNormal, 512, 512, 64, 5, wavelet);
    cudaTransformImpl->reverse3D(input, outputCuda, 512, 512, 64, 5, wavelet);

    std::cout << "Reverse 3D" << std::endl;
    checkError(outputNormal, outputCuda, 512 * 512 * 64);

    int levels = 2;
    elementNum = 16 * 16 * 16;
    wavelet = WaveletFactory::getInstance("haar");
    for(int i = 0; i < elementNum; ++i) {
        input[i] = rand() % 256;
    }

    int x = 16, y = 4, z = 32;

    normalTransformImpl->forward3D(input, outputNormal, x, y, z, levels, wavelet);
    cudaTransformImpl->forward3D(input, outputCuda, x, y, z, levels, wavelet);

    std::cout << "Forward 3D" << std::endl;
    checkError(outputNormal, outputCuda, x * y * z);

    memcpy(input, outputNormal, elementNum);

    normalTransformImpl->reverse3D(input, outputNormal, x, y, z, levels, wavelet);
    cudaTransformImpl->reverse3D(input, outputCuda, x, y, z, levels, wavelet);

    std::cout << "Reverse 3D" << std::endl;
    checkError(outputNormal, outputCuda, x * y * z);
   }


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
