#!/bin/sh

COMPRESSOR='waveletCompressor'
FILES=`find resources -name *.tiff`

mkdir tmp

for FILE in $FILES; do
	echo "Compressing file" $FILE
	FILE_SIMPLE=`basename $FILE .tiff`
	./$COMPRESSOR -ratio 4 $FILE tmp/$FILE_SIMPLE.wci >> tmp/test.log
	if [ ! $? ]; then
		echo Compression FAIL
	fi
	./$COMPRESSOR tmp/$FILE_SIMPLE.wci tmp/$FILE_SIMPLE.tiff >> tmp/test.log
	if [ ! $? ]; then
		echo Decompression FAIL
	fi
done
