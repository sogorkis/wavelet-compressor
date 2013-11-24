/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "CudaEventTimer.h"

CudaEventTimer & CudaEventTimer::start() {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    return *this;
}

CudaEventTimer & CudaEventTimer::stop() {
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    time /= 1000;
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    return *this;
}
