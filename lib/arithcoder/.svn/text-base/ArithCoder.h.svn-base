/*
// Arithmetic coding implementation is based on code from article:
// "Arithmetic Coding revealed" Eric Bodden, Malte Clasen, Joachim Kneis
//
// http://www.sable.mcgill.ca/publications/techreports/2007-5/bodden-07-arithmetic-TR.pdf
//---------------------------------------------------------------------------------*/
#ifndef ARITHCODER_H
#define ARITHCODER_H

#include <iostream>

using namespace std;

class ArithCoder {
public:
        ArithCoder();

        void SetInputFile( istream *stream );
        void SetOutputFile( ostream *stream );

        void Encode( const unsigned int low_count,
                     const unsigned int high_count,
                     const unsigned int total );
        void EncodeFinish();

        void DecodeStart();
        unsigned int DecodeTarget( const unsigned int total );
        void Decode( const unsigned int low_count, const unsigned int high_count );

        unsigned int getEncodeCost(const unsigned int low_count,
                     const unsigned int high_count,
                     const unsigned int total);
protected:
        // bit operations
        void SetBit( const unsigned char bit );
        void SetBitFlush();
        unsigned char GetBit();

        unsigned char mBitBuffer;
        unsigned char mBitCount;

        // in-/output stream
        istream *inStream;
        ostream *outStream;

        // encoder & decoder
        unsigned int mLow;
        unsigned int mHigh;
        unsigned int mStep;
        unsigned int mScale;

        // decoder
        unsigned int mBuffer;
};

#endif // ARITHCODER_H
