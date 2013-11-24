#ifndef TIMERSDATA_H
#define TIMERSDATA_H

#include "CudaEventTimer.h"

#include <string>
#include <map>

using std::string;
using std::map;

class TimersData {
public:
    static TimersData * getInstance();

    void startTimer(string timerName);

    void stopTimer(string timerName);

    float getTimerTime(string timerName);

private:
    TimersData() {}

    static TimersData *instance;

    map<string, float> *timeMap;
    map<string, CudaEventTimer*> *timerMap;
};

#endif // TIMERSDATA_H
