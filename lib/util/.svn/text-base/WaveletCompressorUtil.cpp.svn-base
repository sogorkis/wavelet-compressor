/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include <util/WaveletCompressorUtil.h>

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cmath>
#include <limits>

bool verbose = false;

void setVerbose(bool value) {
    verbose = value;
}

void info(const char *format, ...) {
    if(verbose) {
        va_list args;
        va_start(args,format);
        printf("INFO: ");
        vprintf(format, args);
        printf("\n");
        va_end(args);
    }
}

void fail(const char *format, ...) {
    va_list args;
    va_start(args,format);
    printf("FAIL: ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
    exit(1);
}

void debug(const char *format, ...) {
#ifdef DEBUG
    va_list args;
    va_start(args,format);
    printf("DEBUG: ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
#endif
}
