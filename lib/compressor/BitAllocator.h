/*
// BitAllocator class is based on Allocator class from Wavelet Compression
// Construction Kit.
//
//---------------------------------------------------------------------------
// Baseline Wavelet Transform Coder Construction Kit
//
// Geoff Davis
// gdavis@cs.dartmouth.edu
// http://www.cs.dartmouth.edu/~gdavis
//
// Copyright 1996 Geoff Davis 9/11/96
//
// Permission is granted to use this software for research purposes as
// long as this notice stays attached to this software.
//
//---------------------------------------------------------------------------
// allocator.hh
//
// Given rate/distortion curves for nSets collections of transform
// coefficients (contained in CoeffSet objects), performs a
// constrained optimization of quantizer resolutions.  An array of
// quantizer precisions, precision[i], is found so that the sum (over
// i) of weight[i]*distortion[i][precision[i]] is minimized subject to
// the constraint that the sum (over i) of cost[i][precision[i]] is
// less than or equal to the given budget.
//
// Functions:
// ----------
// optimalAllocate     Does bit allocation using an algorithm described
//                     in Y. Shoham and A. Gersho, "Efficient bit
//                     allocation for an arbitrary set of quantizers,"
//                     IEEE Transactions on Acoustics, Speech, and
//                     Signal Processing, Vol. 36, No. 9,
//                     pp. 1445-1453, Sept 1988.
//
// greedyAugment       The Shoham & Gersho algorithm doesn't yield
//                     optimal allocations for all possible budgets.
//                     The optimalAllocate routine returns the best
//                     allocation that doesn't exceed the given
//                     budget.  GreedyAugment uses marginal analysis
//                     to greedily increase individual quantizer
//                     precisions until we reach the budget.
//                     Allocations will still be a little under budget
//                     but shouldn't be by much.  Note that the header
//                     info is not included in the overall budget.
//
// print               Prints out the current allocation
//
//---------------------------------------------------------------------------*/
#ifndef BITALLOCATOR_H
#define BITALLOCATOR_H

#include "common/SubbandData.h"
#include "arithcoder/ArithCoderModelFactory.h"
#include "quantizer/QuantizerFactory.h"

class BitAllocator {
public:
    BitAllocator();

    virtual ~BitAllocator();

    void calculateRateDistortion(SubbandData &subbandData,
        QuantizerType quantizerType,
        ArithCoderModelType arithCoderModelType);

    void optimalAllocate(int budget, bool augment);

    void greedyAugment(float bitsLeft);

    void allocateLambda(float lambda, float &optimalRate, float &optimalDist);

    void resetPrecision();

    void print();

    float getSubbandMin(int subbandIndex) { return min[subbandIndex]; }

    float getSubbandMax(int subbandIndex) { return max[subbandIndex]; }

    float getSubbandMean(int subbandIndex) { return mean[subbandIndex]; }

    int getOptimalSubbandBits(int subbandIndex) { return precision[subbandIndex]; }

    float getTotalDistortion();

    float getTotalRate();
private:
    void calculateSubbandStatistics(SubbandData &subbandData);

    static const int NQUANT = 10;

    int nSets;
    int *precision, *subbandLengths;
    float ** rate, ** distortion;
    float * min, *max, *mean;
};

#endif // BITALLOCATOR_H
