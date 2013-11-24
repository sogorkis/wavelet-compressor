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
//*/
#include "BitAllocator.h"
#include "quantizer/UniformQuantizer.h"
#include "util/WaveletCompressorUtil.h"
#include "arithcoder/ArithCoderModelOrder0.h"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>

BitAllocator::BitAllocator() {
    precision = NULL;
}

BitAllocator::~BitAllocator() {
    if (precision != NULL) {
        delete [] precision;
    }
    if(rate != NULL) {
        for(int i = 0; i < nSets; ++i) {
            delete [] rate[i];
            delete [] distortion[i];
        }
        delete [] rate;
        delete [] distortion;
        delete [] min;
        delete [] max;
        delete [] mean;
        delete [] subbandLengths;
    }
}

void BitAllocator::resetPrecision() {
  if (precision != NULL)
    delete [] precision;

  precision = new int [nSets];
  for (int i = 0; i < nSets; i++)
    precision[i] = 0;
}

void BitAllocator::calculateRateDistortion(SubbandData &subbandData,
    QuantizerType quantizerType,
    ArithCoderModelType arithCoderModelType) {
    this->nSets = subbandData.getSubbandCount();
    rate = new float*[nSets];
    distortion = new float*[nSets];

    calculateSubbandStatistics(subbandData);

    for(int i = 0; i < nSets; ++i) {
        rate[i] = new float[NQUANT];
        distortion[i] = new float[NQUANT];
    }

    debug("---------Rate distortion calculation-----------");
    debug("| Subband |   bits  |    rate    | distortion |");
    for(int j = 0; j < NQUANT; ++j) {
        ArithCoder arithCoder;
        for(int i = 0; i < nSets; ++i) {
            bool removeMean = i == 0;

            DataIterator *iter = subbandData.getDataIteratorForSubband(i);
            Quantizer *quantizer = QuantizerFactory::getQuantizerInstance(quantizerType, removeMean, min[i], max[i], mean[i]);
            ArithCoderModel *arithModel = ArithCoderModelFactory::getInstance(&arithCoder, arithCoderModelType ,j);

            quantizer->getRateDistortion(*iter, *arithModel, j, rate[i][j], distortion[i][j]);

            debug("|      %3d|      %3d|%12.2f|%12.2f|", i, j, rate[i][j], distortion[i][j]);

            delete iter;
            delete quantizer;
            delete arithModel;
        }
    }
    debug("-----------------------------------------------");
}

void BitAllocator::calculateSubbandStatistics(SubbandData &subbandData) {
    min = new float[nSets];
    max = new float[nSets];
    mean = new float[nSets];
    subbandLengths = new int[nSets];

    debug("--------------------------------------------------");
    debug("| Subband |     min    |     max    |     mean   |");
    for(int i = 0; i < subbandData.getSubbandCount(); ++i) {
        subbandLengths[i] = subbandData.getSubbandLength(i);

        DataIterator * iter = subbandData.getDataIteratorForSubband(i);
        float minValue = std::numeric_limits<float>::max();
        float maxValue = std::numeric_limits<float>::min();
        float meanValue = 0;
        int total = 0;
        while(iter->hasNext()) {
            float value = iter->next();
            meanValue += value;
            if(minValue > value) {
                minValue = value;
            }
            if(maxValue < value) {
                maxValue = value;
            }
            ++total;
        }
        meanValue /= total;
        min[i] = minValue;
        max[i] = maxValue;
        mean[i] = meanValue;
        delete iter;

        debug("|      %3d|%12.2f|%12.2f|%12.2f|", i, minValue, maxValue, meanValue);
    }
    debug("--------------------------------------------------");
}

void BitAllocator::optimalAllocate (int budget, bool augment) {
  float bitBudget = 8 * budget;

  float lambda, lambdaLow, lambdaHigh;
  float rateLow, rateHigh, currentRate;
  float distLow, distHigh, currentDist;

  resetPrecision();

  lambdaLow = 0.0;
  allocateLambda (lambdaLow, rateLow, distLow);
  if (rateLow < bitBudget) // this uses the largest possible # of bits
    return;                //   -- if this is within the budget, do it

  lambdaHigh = 1000000.0;
  float lastRateHigh = -1;
  do {
    // try to use the smallest possible # of bits
    allocateLambda(lambdaHigh, rateHigh, distHigh);

    // if this is still > bitBudget, try again w/ larger lambda
    if (rateHigh > bitBudget && lastRateHigh != rateHigh) {
      lambdaLow = lambdaHigh;
      rateLow = rateHigh;
      distLow = distHigh;
      lambdaHigh *= 10.0;
    }
  } while (rateHigh > bitBudget && lastRateHigh != rateHigh);

  // give up when changing lambda has no effect on things
  if (lastRateHigh == rateHigh)
    return;

  // Note rateLow will be > rateHigh
  if (rateLow < bitBudget)
    fail("Failed to bracket bit budget = %d: rateLow = %g rateHigh = %g\n", budget, rateLow, rateHigh);

  while (lambdaHigh - lambdaLow > 0.01)  {
    lambda = (lambdaLow + lambdaHigh)/2.0;

    allocateLambda(lambda, currentRate, currentDist);

    if (currentRate > bitBudget)
      lambdaLow = lambda;
    else
      lambdaHigh = lambda;
  }

  if (currentRate > bitBudget)  {
    lambda = lambdaHigh;
    allocateLambda(lambda, currentRate, currentDist);
  }

  if (augment)
    greedyAugment(bitBudget-currentRate);
}

void BitAllocator::allocateLambda(float lambda, float &optimalRate, float &optimalDist) {
   int i, j;
   float G, minG, minRate, minDist;

   optimalRate = optimalDist = 0.0;

   // want to minimize G = distortion + lambda * rate

   // loop through all rate-distortion curves
   for (i = 0; i < nSets; i++)  {
     minG = minRate = minDist = FLT_MAX;

     for (j = 0; j < NQUANT; j++) {
       G = distortion[i][j] + lambda * rate[i][j];
       if (G < minG)  {
         minG = G;
         minRate = rate[i][j];
         minDist = distortion[i][j];
         precision[i] = j;
       }
     }

     optimalRate += minRate;
     optimalDist += minDist;
   }
//   debug("lambda = %g  optimal rate = %g, optimal dist = %g", lambda, optimalRate, optimalDist);
}

void BitAllocator::greedyAugment(float bitsLeft) {
  int bestSet, newPrecision = -1;
  float delta, maxDelta, bestDeltaDist, bestDeltaRate = 0;

  do {
    bestSet = -1;
    maxDelta = 0;

    // Find best coeff set to augment
    for (int i = 0; i < nSets; i++) {
      for (int j = precision[i]+1; j < NQUANT; j++) {
        float deltaRate = rate[i][j] - rate[i][precision[i]];
        float deltaDist = -(distortion[i][j] - distortion[i][precision[i]]);

        if (deltaRate != 0 && deltaRate <= bitsLeft) {
          delta = deltaDist / deltaRate;

          if (delta > maxDelta) {
            maxDelta = delta;
            bestDeltaRate = deltaRate;
            bestDeltaDist = deltaDist;
            bestSet = i;
            newPrecision = j;
          }
        }
      }
    }

    if (bestSet != -1) {
      precision[bestSet] = newPrecision;
      bitsLeft -= bestDeltaRate;
    }
  } while (bestSet != -1);
}

void BitAllocator::print()
{
  float totalRate = 0, totalDist = 0;
  int totalData = 0;

  info("---------- Rate distortion optimal values--------------------");
  info("| Set | Bits |     Rate             |       Distortion");
  for (int i = 0; i < nSets; i++) {
    info("| %2d  |  %2d  | %12.2f | %5.2f |  %12.2f   %7.2f",
            i, precision[i],
            rate[i][precision[i]],
            rate[i][precision[i]]/(float)subbandLengths[i],
            distortion[i][precision[i]],
            distortion[i][precision[i]]/(float)subbandLengths[i]);
    totalRate += rate[i][precision[i]];
    totalDist += distortion[i][precision[i]];
    totalData += subbandLengths[i];
  }
  float rms = sqrt(totalDist/(float)totalData);
  float psnr = 20.0 * log(255.0/rms)/log(10.0);
  info("----------------------Summary--------------------------------");
  info("total rate              = %g", totalRate/8.0);
  info("total dist              = %g", totalDist);
  info("total coeffs            = %d", totalData);
  info("RMS error               = %g", rms);
  info("PSNR (transform domain) = %g", psnr);
  info("-------------------------------------------------------------");
}

float BitAllocator::getTotalDistortion() {
    float totalDist = 0;
    for (int i = 0; i < nSets; i++) {
        totalDist += distortion[i][precision[i]];
    }
    return totalDist;
}

float BitAllocator::getTotalRate() {
    float totalRate = 0;
    for (int i = 0; i < nSets; i++) {
        totalRate += rate[i][precision[i]];
    }
    return totalRate;
}
