/*
// Arithmetic coding implementation is based on code from article:
// "Arithmetic Coding revealed" Eric Bodden, Malte Clasen, Joachim Kneis
//
// http://www.sable.mcgill.ca/publications/techreports/2007-5/bodden-07-arithmetic-TR.pdf
//---------------------------------------------------------------------------------*/
#ifndef ARITHCODERMODEL_H
#define ARITHCODERMODEL_H

#include "ArithCoder.h"

enum ModeE {
    MODE_ENCODE = 0,
    MODE_DECODE
};

class ArithCoderModel {
public:
    ArithCoderModel(ArithCoder *mAC) {
        this->mAC = mAC;
    }

    virtual ~ArithCoderModel() {}

    void setEncodeStream(ostream *stream);

    void setDecodeStream(istream *stream);

    virtual void EncodeSymbol(unsigned int symbol) = 0;

    virtual void FinishEncode() = 0;

    virtual int DecodeSymbol() = 0;

    virtual unsigned int getEncodeCost(unsigned int symbol) = 0;

    void encodeFloat(float value);

    void encodeUChar(unsigned char value);

    float decodeFloat();

    unsigned char decodeUChar();
protected:
    ArithCoder *mAC;
    istream *mSource;
    ostream *mTarget;
};

#endif // ARITHCODERMODEL_H
