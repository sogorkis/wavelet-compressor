/*
// Arithmetic coding implementation is based on code from article:
// "Arithmetic Coding revealed" Eric Bodden, Malte Clasen, Joachim Kneis
//
// http://www.sable.mcgill.ca/publications/techreports/2007-5/bodden-07-arithmetic-TR.pdf
//---------------------------------------------------------------------------------*/

#include "ArithCoderModelOrder0.h"
#include <math.h>

ArithCoderModelOrder0::ArithCoderModelOrder0(ArithCoder *mAC, int bits)
    : ArithCoderModel(mAC) {
    this->escapeSymbol = pow(2, bits);

    // initialize probabilities with 1
    mTotal = escapeSymbol + 1; // 256 + escape symbol for termination
    mCumCount = new unsigned int[mTotal];
    for( unsigned int i=0; i<mTotal; i++ ) {
        mCumCount[i] = 1;
    }
}

ArithCoderModelOrder0::~ArithCoderModelOrder0() {
    delete [] mCumCount;
}

void ArithCoderModelOrder0::EncodeSymbol(unsigned int symbol) {
    // cumulate frequencies
    unsigned int low_count = 0;
    unsigned int j=0;
    for( ; j<symbol; j++ )
        low_count += mCumCount[j];

    // encode symbol
    mAC->Encode( low_count, low_count + mCumCount[j], mTotal );

    // update model
    mCumCount[ symbol ]++;
    mTotal++;
}

unsigned int ArithCoderModelOrder0::getEncodeCost(unsigned int symbol) {
    // cumulate frequencies
    unsigned int bitCost;
    unsigned int low_count = 0;
    unsigned int j=0;
    for( ; j<symbol; j++ )
        low_count += mCumCount[j];

    // encode symbol
    bitCost = mAC->getEncodeCost( low_count, low_count + mCumCount[j], mTotal );

    // update model
    mCumCount[ symbol ]++;
    mTotal++;

    return bitCost;
}

void ArithCoderModelOrder0::FinishEncode() {
    mAC->Encode( mTotal-1, mTotal, mTotal );
    mAC->EncodeFinish();
}

int ArithCoderModelOrder0::DecodeSymbol() {
    unsigned int symbol;

    // read value
    unsigned int value = mAC->DecodeTarget( mTotal );

    unsigned int low_count = 0;

    // determine symbol
    for( symbol=0; low_count + mCumCount[symbol] <= value; symbol++ ) {
        low_count += mCumCount[symbol];
    }

    // adapt decoder
    mAC->Decode( low_count, low_count + mCumCount[ symbol ] );

    // update model
    mCumCount[ symbol ]++;
    mTotal++;

    return symbol != escapeSymbol ? symbol : -1;
}
