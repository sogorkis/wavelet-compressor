#include "HostEventTimer.h"

#define BILLION  1000000000L;

HostEventTimer& HostEventTimer::start() {
    clock_gettime(CLOCK_MONOTONIC, &startTimespec);
    return *this;
}

HostEventTimer& HostEventTimer::stop() {
    clock_gettime(CLOCK_MONOTONIC, &endTimespec);
    return *this;
}

double HostEventTimer::getTime() {
    return  (endTimespec.tv_sec - startTimespec.tv_sec) + (double)(endTimespec.tv_nsec - startTimespec.tv_nsec) / (double)BILLION;
}
