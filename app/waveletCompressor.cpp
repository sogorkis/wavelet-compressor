/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <sys/stat.h>

#include "color_transform/YCbCrColorTransform.h"
#include "wavelet_transform/WaveletFactory.h"
#include "compressor/CudaWaveletCompressor.h"
#include "common/OpenCvImage.h"
#include "common/OpenCvImageSequence.h"
#include "util/TimersData.h"

const char* AVI_EXTENSION = "avi";
const char* PGM_EXTENSION = "pgm";
const char* BMP_EXTENSION = "bmp";
const char* JPG_EXTENSION = "jpg";
const char* TIFF_EXTENSION = "tiff";
const char* WCV_EXTENSION = "wcv";
const char* WCI_EXTENSION = "wci";

const char* USAGE =
"\nOPTIONS:\n"
"-help              - prints this help\n"
"-verbose           - verbose output\n"
"-cuda              - uses cuda implementation\n"
"-wavelet <name>    - specify wavelet (haar, daub4, daub6, cdf97, antonini); compression only; default antonini\n"
"-ratio <value>     - specify compression ratio; compression only; required\n"
"-quantizer <name>  - specify quantizer type (utq, dutq); compression only; default dutq (deadzone uniform quantizer)\n"
"-levels <value>    - specify number of decomposition levels; compression only; default 5\n";

using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::fstream;

int  getArgumentIndex(int argc, char *argv[], string argument);
void printUsage(string &programName);
void fatal(string &programName, string errorMessage);
void checkCompressionArgs(string &programName, int ratio, int ratioIndex, int levels);
int  getFileSize(string fileName);
void printCompressInfo(int channelCount, int rawSize, float targetRatio, string fileName, WaveletCompressor *compressor);

void compressVideo(string inputFileName, string outputFileName, bool cuda, int ratio, int levels, string &waveletName,
                   QuantizerType quantizerType, bool verbose);
void decompressVideo(string inputFileName, string outputFileName, bool cuda, bool verbose);

void compressImage(string inputFileName, string outputFileName, bool cuda, int ratio, int levels, string &waveletName,
                   QuantizerType quantizerType, bool verbose);
void decompressImage(string inputFileName, string outputFileName, bool cuda, bool verbose);


int main(int argc, char *argv[]) {
    string programName(argv[0]);
    int s = programName.find_last_of("\\/");
    programName = programName.substr(s + 1, programName.length() - s - 1);

    if(argc < 3) {
        printUsage(programName);
        return 1;
    }

    if(getArgumentIndex(argc, argv, "-help") > 0) {
        printUsage(programName);
        return 0;
    }

    string fileName1(argv[argc - 2]);
    string fileName2(argv[argc - 1]);
    int s1 = fileName1.find_last_of("\\.");
    int s2 = fileName2.find_last_of("\\.");
    string extension1 = fileName1.substr(s1 + 1, fileName1.length() - s1 - 1);
    string extension2 = fileName2.substr(s2 + 1, fileName2.length() - s2 - 1);

    bool cuda           = getArgumentIndex(argc, argv, "-cuda") > 0;
    bool verbose        = getArgumentIndex(argc, argv, "-verbose") > 0;
    int waveletIndex    = getArgumentIndex(argc, argv, "-wavelet") + 1;
    int ratioIndex      = getArgumentIndex(argc, argv, "-ratio") + 1;
    int quantizerIndex  = getArgumentIndex(argc, argv, "-quantizer") + 1;
    int levelsIndex     = getArgumentIndex(argc, argv, "-levels") + 1;
    string waveletName  = "antonini";
    int ratio           = -1;
    int levels          = 5;
    QuantizerType quantizerType = DEADZONE_UNIFORM_QUANTIZER;

    if(ratioIndex > 0 && ratioIndex < argc) {
        ratio = atoi(argv[ratioIndex]);
    }

    bool levelsDefined = levelsIndex > 0 && levelsIndex < argc;
    if(levelsDefined) {
        levels = atoi(argv[levelsIndex]);
    }

    if(waveletIndex > 0 && ratioIndex < argc) {
        waveletName = argv[waveletIndex];
        bool correct = false;
        for(int i = 0; i < WaveletFactory::waveletNamesLength; ++i) {
            if(waveletName == WaveletFactory::waveletNames[i]) {
                correct = true;
            }
        }

        if(!correct) {
            fatal(programName, "Invalid wavelet name: " + waveletName);
        }
    }

    if(quantizerIndex > 0 && ratioIndex < argc) {
        string quantizerString = argv[quantizerIndex];
        if(quantizerString == "dutq") {
            // do nothing
        }
        else if (quantizerString == "utq") {
            quantizerType = UNIFORM_QUANTIZER;
        }
        else {
            fatal(programName, "Invalid quantizer type: " + quantizerString);
        }
    }

    setVerbose(verbose);
    if(extension1 == AVI_EXTENSION && extension2 == WCV_EXTENSION) {
        if (!levelsDefined) {
            levels = 3;
        }
        checkCompressionArgs(programName, ratio, ratioIndex, levels);
        compressVideo(fileName1, fileName2, cuda, ratio, levels, waveletName, quantizerType, verbose);
    }
    else if(extension2 == AVI_EXTENSION && extension1 == WCV_EXTENSION) {
        decompressVideo(fileName1, fileName2, cuda, verbose);
    }
    else if((extension1 == PGM_EXTENSION || extension1 == BMP_EXTENSION || extension1 == JPG_EXTENSION
             || extension1 == TIFF_EXTENSION) && extension2 == WCI_EXTENSION) {
        checkCompressionArgs(programName, ratio, ratioIndex, levels);
        compressImage(fileName1, fileName2, cuda, ratio, levels, waveletName, quantizerType, verbose);
    }
    else if((extension2 == PGM_EXTENSION || extension2 == BMP_EXTENSION || extension2 == JPG_EXTENSION
             || extension2 == TIFF_EXTENSION) && extension1 == WCI_EXTENSION) {
        decompressImage(fileName1, fileName2, cuda, verbose);
    }
    else {
        fatal(programName, "Bad args");
    }

    return 0;
}

int getArgumentIndex(int argc, char *argv[], string argument) {
    for(int i = 0; i < argc; ++i) {
        if(argument == argv[i]) {
            return i;
        }
    }
    return -1;
}

void printUsage(string &programName) {
    cerr << "Usage: " << programName << " [OPTIONS] <input file name> <output file name>" << endl;
    cerr << USAGE << endl;
}

void fatal(string &programName, string errorMessage) {
    cerr << errorMessage << endl;
    printUsage(programName);
    exit(1);
}

void checkCompressionArgs(string &programName, int ratio, int ratioIndex, int levels) {
    if(ratioIndex == 0) {
        fatal(programName, "No ratio given");
    }
    if(!(ratio > 0 && ratio <= 256)) {
        fatal(programName, "Invalid ratio value: " + ratio);
    }
    if(!(levels > 0)) {
        fatal(programName, "Invalid levels value: " + levels);
    }
}

void compressImage(string inputFileName, string outputFileName, bool cuda, int ratio, int levels, string &waveletName,
                   QuantizerType quantizerType, bool verbose) {
    TimersData::getInstance()->startTimer("openImage");
    IplImage* img = cvLoadImage(inputFileName.c_str(), -1);
    TimersData::getInstance()->stopTimer("openImage");

    if(img == NULL) {
        cerr << "Invalid file name: " << inputFileName << endl;
        exit(1);
    }
    OpenCvImage cvImage(img);
    ColorTransformType colorTransformType = EMPTY_COLOR_TRANSFORM;
    if (cvImage.getChannelCount() > 1) {
        colorTransformType = YCBCR_COLOR_TRANSFORM;
    }

    Wavelet * wavelet = WaveletFactory::getInstance(waveletName);

    std::fstream encodeFile(outputFileName.c_str(), std::ios::binary | std::ios::out);
    WaveletCompressor *compressor;
    if(!cuda) {
        compressor = new WaveletCompressor();
    }
    else {
        compressor = new CudaWaveletCompressor();
    }

    info("Compressing file %s", inputFileName.c_str());
    compressor->encode(&cvImage, wavelet, ratio, &encodeFile, levels, quantizerType, MODEL_ORDER0,
                      colorTransformType);

    encodeFile.close();
    info("Compression complete. Output file %s", outputFileName.c_str());

    int channelCount = cvImage.getChannelCount();
    int rawSize = cvImage.getWidth() * cvImage.getHeight() * channelCount;
    printCompressInfo(channelCount, rawSize, ratio, outputFileName, compressor);

    info("--------------------------------------------------------------");
    info("Used time:");
    info("OpenCV open image             = %f s", TimersData::getInstance()->getTimerTime("openImage"));
    if (cuda) {
        info("CUDA malloc time              = %f s", TimersData::getInstance()->getTimerTime("cudaMalloc"));
        info("CUDA copy host -> device time = %f s", TimersData::getInstance()->getTimerTime("cudaHostDevice"));
        info("CUDA color transform time     = %f s", TimersData::getInstance()->getTimerTime("cudaColorTransform"));
        info("CUDA transform time           = %f s", TimersData::getInstance()->getTimerTime("cudaTransform"));
        info("CUDA copy device -> host      = %f s", TimersData::getInstance()->getTimerTime("cudaDeviceHost"));
    }
    else {
        info("Color transform time          = %lf s", TimersData::getInstance()->getTimerTime("colorTransform"));
        info("Transform time                = %lf s", TimersData::getInstance()->getTimerTime("transform"));
    }
    info("Linearize time                = %f s", TimersData::getInstance()->getTimerTime("linearize"));
    info("RD optimize time              = %f s", TimersData::getInstance()->getTimerTime("rdOptimize"));
    info("Quantize and encode time      = %f s", TimersData::getInstance()->getTimerTime("quantizeEncode"));

    cvReleaseImage(&img);
    delete wavelet;
    delete compressor;
}

void decompressImage(string inputFileName, string outputFileName, bool cuda, bool verbose) {
    std::fstream decodeFile(inputFileName.c_str(), std::ios::binary | std::ios::in);
    WaveletCompressor *compressor;
    if(!cuda) {
        compressor = new WaveletCompressor();
    }
    else {
        compressor = new CudaWaveletCompressor();
    }

    OpenCvImage cvImage;

    info("Decompressing file %s", inputFileName.c_str());
    compressor->decode(&decodeFile, &cvImage);

    TimersData::getInstance()->startTimer("saveImage");
    if(!cvSaveImage(outputFileName.c_str(), cvImage.getIplImage())) {
        cerr << "Could not save: " << outputFileName << endl;
    }
    TimersData::getInstance()->stopTimer("saveImage");

    info("Decompression complete. Output file %s", outputFileName.c_str());
    info("--------------------------------------------------------------");
    info("Used time:");
    info("OpenCV save image             = %f s", TimersData::getInstance()->getTimerTime("saveImage"));
    if (cuda) {
        info("CUDA malloc time              = %f s", TimersData::getInstance()->getTimerTime("cudaMalloc"));
        info("CUDA copy host -> device time = %f s", TimersData::getInstance()->getTimerTime("cudaHostDevice"));
        info("CUDA color transform time     = %f s", TimersData::getInstance()->getTimerTime("cudaColorTransform"));
        info("CUDA transform time           = %f s", TimersData::getInstance()->getTimerTime("cudaTransform"));
        info("CUDA copy device -> host      = %f s", TimersData::getInstance()->getTimerTime("cudaDeviceHost"));
    }
    else {
        info("Color transform time          = %lf s", TimersData::getInstance()->getTimerTime("colorTransform"));
        info("Transform time                = %lf s", TimersData::getInstance()->getTimerTime("transform"));
    }
    info("Delinearize time              = %f s", TimersData::getInstance()->getTimerTime("delinearize"));
    info("Deqantize and decode time     = %f s", TimersData::getInstance()->getTimerTime("dequantizeDecode"));

    delete compressor;
}

void compressVideo(string inputFileName, string outputFileName, bool cuda, int ratio, int levels, string &waveletName,
                   QuantizerType quantizerType, bool verbose) {
    OpenCvImageSequence imageSequence;
    imageSequence.initReadSequence(inputFileName);
    ColorTransformType colorTransformType = EMPTY_COLOR_TRANSFORM;
    if (imageSequence.getChannelCount() > 1) {
        colorTransformType = YCBCR_COLOR_TRANSFORM;
    }

    cout << "--------------------" << endl
            << "Video properties:" << endl
            << "height:" << imageSequence.getHeight() << endl
            << "width:" << imageSequence.getWidth() << endl
            << "fps:" << imageSequence.getFps() << endl
            << "frame count:" << imageSequence.getFrameCount() << endl << endl;

    Wavelet * wavelet = WaveletFactory::getInstance(waveletName);

    fstream encodeFile(outputFileName.c_str(), std::ios::binary | std::ios::out);
    WaveletCompressor *compressor;
    if(!cuda) {
        compressor = new WaveletCompressor();
    }
    else {
        compressor = new CudaWaveletCompressor();
    }

    info("Compressing file %s", inputFileName.c_str());
    compressor->encode(&imageSequence, wavelet, ratio, &encodeFile, levels, quantizerType, MODEL_ORDER0,
                      colorTransformType);

    encodeFile.close();
    imageSequence.close();
    info("Compression complete. Output file %s", outputFileName.c_str());

    int channelCount = imageSequence.getChannelCount();
    int rawSize = imageSequence.getWidth() * imageSequence.getHeight() * imageSequence.getFrameCount() * channelCount;
    printCompressInfo(channelCount, rawSize, ratio, outputFileName, compressor);

    info("--------------------------------------------------------------");
    info("Used time:");
    info("OpenCV open image sequence    = %f s", TimersData::getInstance()->getTimerTime("openImageSequence"));
    if (cuda) {
        info("CUDA malloc time              = %f s", TimersData::getInstance()->getTimerTime("cudaMalloc"));
        info("CUDA copy host -> device time = %f s", TimersData::getInstance()->getTimerTime("cudaHostDevice"));
        info("CUDA color transform time     = %f s", TimersData::getInstance()->getTimerTime("cudaColorTransform"));
        info("CUDA transform time           = %f s", TimersData::getInstance()->getTimerTime("cudaTransform"));
        info("CUDA copy device -> host      = %f s", TimersData::getInstance()->getTimerTime("cudaDeviceHost"));
    }
    else {
        info("Color transform time          = %lf s", TimersData::getInstance()->getTimerTime("colorTransform"));
        info("Transform time                = %lf s", TimersData::getInstance()->getTimerTime("transform"));
    }
    info("Linearize time                = %f s", TimersData::getInstance()->getTimerTime("linearize"));
    info("RD optimize time              = %f s", TimersData::getInstance()->getTimerTime("rdOptimize"));
    info("Quantize and encode time      = %f s", TimersData::getInstance()->getTimerTime("quantizeEncode"));

    delete wavelet;
    delete compressor;
}

void decompressVideo(string inputFileName, string outputFileName, bool cuda, bool verbose) {
    std::fstream decodeFile(inputFileName.c_str(), std::ios::binary | std::ios::in);

    OpenCvImageSequence imageSequence;
    imageSequence.setWriteSequenceFileName(outputFileName);

    WaveletCompressor *compressor;
    if(!cuda) {
        compressor = new WaveletCompressor();
    }
    else {
        compressor = new CudaWaveletCompressor();
    }

    info("Decompressing file %s", inputFileName.c_str());
    compressor->decode(&decodeFile, &imageSequence);

    imageSequence.close();
    decodeFile.close();
    info("Decompression complete. Output file %s", outputFileName.c_str());
    info("--------------------------------------------------------------");
    info("Used time:");
    info("OpenCV save image sequence    = %f s", TimersData::getInstance()->getTimerTime("saveImageSequence"));
    if (cuda) {
        info("CUDA malloc time              = %f s", TimersData::getInstance()->getTimerTime("cudaMalloc"));
        info("CUDA copy host -> device time = %f s", TimersData::getInstance()->getTimerTime("cudaHostDevice"));
        info("CUDA color transform time     = %f s", TimersData::getInstance()->getTimerTime("cudaColorTransform"));
        info("CUDA transform time           = %f s", TimersData::getInstance()->getTimerTime("cudaTransform"));
        info("CUDA copy device -> host      = %f s", TimersData::getInstance()->getTimerTime("cudaDeviceHost"));
    }
    else {
        info("Color transform time          = %lf s", TimersData::getInstance()->getTimerTime("colorTransform"));
        info("Transform time                = %lf s", TimersData::getInstance()->getTimerTime("transform"));
    }
    info("Delinearize time              = %f s", TimersData::getInstance()->getTimerTime("delinearize"));
    info("Deqantize and decode time     = %f s", TimersData::getInstance()->getTimerTime("dequantizeDecode"));

    delete compressor;
}

int getFileSize(string fileName) {
    struct stat filestatus;
    stat(fileName.c_str(), &filestatus);
    return filestatus.st_size;
}

void printCompressInfo(int channelCount, int rawSize, float targetRatio, string fileName, WaveletCompressor *compressor) {
    int fileSize = getFileSize(fileName);
    float obtainedRatio = (float) rawSize / fileSize;
    info("--------------------------------------------------------------");
    info("Raw size             = %d", rawSize);
    info("Obtained size        = %d", fileSize);
    info("Obtained ratio       = %f (target = %f)", obtainedRatio, targetRatio);
    info("Bitrate              = %f", 8 / obtainedRatio);
    if (channelCount > 1) {
        info("Distortion per channel");
        for (int i = 0; i < channelCount; ++i) {
            info("Distortion channel %d = %f", i, compressor->getTotalDistortion(i));
        }
    }
    info("Total distortion     = %f", compressor->getTotalDistortion());
    if (channelCount > 1) {
        info("PSNR per channel");
        for (int i = 0; i < channelCount; ++i) {
            float rms = sqrt(compressor->getTotalDistortion(i)/(rawSize / channelCount));
            float psnr = 20.0 * log(255.0/rms)/log(10.0);
            info("PSNR channel %d       = %f", i, psnr);
        }
    }
    float rms = sqrt(compressor->getTotalDistortion()/rawSize);
    float psnr = 20.0 * log(255.0/rms)/log(10.0);
    info("Total PSNR           = %f", psnr);
}
