/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef CUDAEVENTTIMER_H
#define CUDAEVENTTIMER_H

#include <cuda_runtime.h>

/**
  * Class responsible for GPU time measurement. It's based on cudaEvent_t and it's basic usage
  * should be:
  *
  * <pre>
  * CudaEventTimer timer;
  * timer.start();
  *
  * // things to be measured
  *
  * float time = timer.stop().getTime().
  * </pre>
  *
  * Already used timer can be reused without any special treatment.
  */
class CudaEventTimer
{
public:
    /**
      * Starts timer.
      */
    CudaEventTimer & start();

    /**
      * Stops timer.
      */
    CudaEventTimer & stop();

    /**
      * Returns time between start() and stop() calls.
      * @return time between start() and stop() calls.
      */
    float getTime() { return time; }
private:
    cudaEvent_t startEvent, stopEvent;
    float time;
};

#endif // CUDAEVENTTIMER_H
