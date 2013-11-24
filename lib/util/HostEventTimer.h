#ifndef HOSTEVENTTIMER_H
#define HOSTEVENTTIMER_H

#include <time.h>

class HostEventTimer {
public:
    /**
      * Starts timer.
      */
    HostEventTimer& start();

    /**
      * Stops timer.
      */
    HostEventTimer& stop();

    /**
      * Returns time between start() and stop() calls.
      * @return time between start() and stop() calls.
      */
    double getTime();
private:
    timespec startTimespec, endTimespec;
};

#endif // HOSTEVENTTIMER_H
