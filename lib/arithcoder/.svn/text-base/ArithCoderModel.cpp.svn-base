/*
// Arithmetic coding implementation is based on code from article:
// "Arithmetic Coding revealed" Eric Bodden, Malte Clasen, Joachim Kneis
//
// http://www.sable.mcgill.ca/publications/techreports/2007-5/bodden-07-arithmetic-TR.pdf
//---------------------------------------------------------------------------------*/

#include "ArithCoderModel.h"

void ArithCoderModel::setEncodeStream(ostream *stream) {
    mTarget = stream;
    mAC->SetOutputFile(mTarget);
}

void ArithCoderModel::setDecodeStream(istream *stream) {
    mSource = stream;
    mAC->SetInputFile(mSource);
    mAC->DecodeStart();
}

void ArithCoderModel::encodeFloat(float value) {
    for(unsigned int i = 0; i < sizeof(float); ++i) {
        unsigned char byte = *(reinterpret_cast<unsigned char *>(&value) + i);
        EncodeSymbol(byte);
    }
}

void ArithCoderModel::encodeUChar(unsigned char value) {
    EncodeSymbol(value);
}

float ArithCoderModel::decodeFloat() {
    float ret;
    for(unsigned int i = 0; i < sizeof(float); ++i) {
        *(reinterpret_cast<unsigned char *>(&ret) + i) = DecodeSymbol();
    }
    return ret;
}

unsigned char ArithCoderModel::decodeUChar() {
    return DecodeSymbol();
}
