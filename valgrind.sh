#!/bin/sh

valgrind --tool=memcheck --leak-check=yes --error-limit=yes --suppressions=suppress.valgrind --track-origins=yes ./waveletCompressor encoded.wcv reversed.avi 2> valgrind.log
