/*
// Arithmetic coding implementation is based on code from article:
// "Arithmetic Coding revealed" Eric Bodden, Malte Clasen, Joachim Kneis
//
// http://www.sable.mcgill.ca/publications/techreports/2007-5/bodden-07-arithmetic-TR.pdf
//---------------------------------------------------------------------------------*/
#ifndef ARITHCODERMODELORDER0_H
#define ARITHCODERMODELORDER0_H

#include "ArithCoderModel.h"

#include <iostream>

class ArithCoderModelOrder0 : public ArithCoderModel {
public:
        ArithCoderModelOrder0(ArithCoder *mAC, int bits);

        virtual ~ArithCoderModelOrder0();

        virtual void EncodeSymbol(unsigned int symbol);

        virtual unsigned int getEncodeCost(unsigned int symbol);

        virtual void FinishEncode();

        virtual int DecodeSymbol();
protected:
        unsigned int escapeSymbol;
        unsigned int *mCumCount;
        unsigned int mTotal;
};

#endif // ARITHCODERMODELORDER0_H
