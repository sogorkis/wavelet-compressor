#include <gtest/gtest.h>
#include <math.h>

#include "quantizer/UniformQuantizer.h"
#include "quantizer/DeadzoneUniformQuantizer.h"

class QuantizerTest : public ::testing::Test {
public:
    template<class T, class E>
    static bool checkArraysEqual(const T * expected, const E * actual, int length, float epsylon) {
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
};

TEST_F(QuantizerTest, TestUniformQuantizerQuantizeAndDequantize) {
    int elementNum = 255;
    float input[elementNum];
    float dequantized[elementNum];
    unsigned int quantized[elementNum];
    for(int i = 0; i < elementNum; ++i) {
        input[i] = i;
    }

    UniformQuantizer quantizer(false, 0, elementNum-1, 0, 8);
    quantizer.quantize(input, quantized, elementNum);
    quantizer.dequantize(quantized, dequantized, elementNum);
    bool ret = checkArraysEqual<float, float>(input, dequantized, elementNum, 0.5);
    ASSERT_TRUE(ret);
}

TEST_F(QuantizerTest, TestDeadzoneUniformQuantizerQuantizeSimple) {
    int bits = 4;
    int elementNum = 4;
    float input[] = {0, -2, 2.5, -7};

    unsigned int quantized[elementNum];
    float dequantized[elementNum];

    DeadzoneUniformQuantizer quantizer(false, -7, 2.5, 0, bits);
    quantizer.quantize(input, quantized, elementNum);

    unsigned int maxQuantized = (1 << bits) - 1;
    for(int i = 0; i < elementNum; ++i) {
        ASSERT_FALSE(quantized[i] > maxQuantized) << "quantized: " << quantized[i];
    }

    quantizer.dequantize(quantized, dequantized, elementNum);
    bool ret = checkArraysEqual<float, float>(input, dequantized, elementNum, 0.5);
    ASSERT_TRUE(ret);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
