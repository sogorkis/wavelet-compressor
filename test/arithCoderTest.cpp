#include <gtest/gtest.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <vector>

//#define DEBUG_TEST

#include "arithcoder/ArithCoderModelOrder0.h"

class ArithCoderTest : public ::testing::Test {
public:
    ::testing::AssertionResult testFile(const char * path) {
        std::fstream inputFile(path, std::ios::binary | std::ios::in);
        std::stringstream strEncoded(std::ios::binary | std::ios::in | std::ios::out);
        ArithCoder arithCoder1, arithCoder2;
        ArithCoderModelOrder0 modelEncode(&arithCoder1, 8), modelDecode(&arithCoder2, 8);
        modelEncode.setEncodeStream(&strEncoded);
        std::vector<unsigned char> data;

        while(!inputFile.eof()) {
            unsigned char ch;
            inputFile.read(reinterpret_cast<char *>(&ch), sizeof(char));
            data.push_back(ch);
            modelEncode.EncodeSymbol(ch);
        }
        modelEncode.FinishEncode();
        inputFile.close();

        modelDecode.setDecodeStream(&strEncoded);
        int i = 0, elementNum = data.size();
        int e;
        while((e = modelDecode.DecodeSymbol()) >= 0) {
            if(i >= elementNum) {
                ::testing::Message msg;
                msg << "i >= elementNum";
                return ::testing::AssertionFailure(msg);
            }

            if(data[i++] != e) {
                ::testing::Message msg;
                msg << "data[i++] != e";
                return ::testing::AssertionFailure(msg);
            }
        }

        if(i != elementNum) {
            ::testing::Message msg;
            msg << "i != elementNum";
            return ::testing::AssertionFailure(msg);
        }

        return ::testing::AssertionSuccess();
    }
};

TEST_F(ArithCoderTest, TestModelOrder0Files) {
    ASSERT_TRUE(testFile("resources/calgary/geo"));
    ASSERT_TRUE(testFile("resources/calgary/bib"));
    ASSERT_TRUE(testFile("resources/calgary/news"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

