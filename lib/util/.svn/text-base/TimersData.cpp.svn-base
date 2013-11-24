#include "TimersData.h"

TimersData *TimersData::instance = NULL;

TimersData * TimersData::getInstance() {
    if (instance == NULL) {
        instance = new TimersData();
        instance->timeMap = new map<string, float>();
        instance->timerMap = new map<string, CudaEventTimer*>();
    }
    return instance;
}

void TimersData::startTimer(string timerName) {
    map<string, CudaEventTimer*>::iterator it = timerMap->find(timerName);
    if (it == timerMap->end()) {
        (*timerMap)[timerName] = new CudaEventTimer();
        (*timeMap)[timerName] = 0;
    }
    (*timerMap)[timerName]->start();
}

void TimersData::stopTimer(string timerName) {
    (*timeMap)[timerName] += (*timerMap)[timerName]->stop().getTime();
}

float TimersData::getTimerTime(string timerName) {
    return (*timeMap)[timerName];
}
