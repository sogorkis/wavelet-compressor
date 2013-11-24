HEADERS += lib/wavelet_transform/Wavelet.h \
    lib/wavelet_transform/WaveletTransform.h \
    lib/wavelet_transform/WaveletFactory.h \
    lib/wavelet_transform/CudaWaveletTransform.h \
    lib/util/WaveletCompressorUtil.h \
    lib/wavelet_transform/WaveletTransformImpl.h \
    lib/wavelet_transform/WaveletTransformCudaImpl.h \
    lib/util/CudaEventTimer.h \
    lib/wavelet_transform/Filter.h \
    lib/quantizer/Quantizer.h \
    lib/quantizer/UniformQuantizer.h \
    lib/common/Image.h \
    lib/common/OpenCvImage.h \
    lib/compressor/WaveletCompressor.h \
    lib/compressor/BitAllocator.h \
    lib/arithcoder/ArithCoder.h \
    lib/arithcoder/ArithCoderModel.h \
    lib/arithcoder/ArithCoderModelOrder0.h \
    lib/common/SubbandData.h \
    lib/common/DataIterator.h \
    lib/quantizer/QuantizerFactory.h \
    lib/arithcoder/ArithCoderModelFactory.h \
    lib/common/ImageSequence.h \
    lib/common/OpenCvImageSequence.h \
    lib/quantizer/DeadzoneUniformQuantizer.h \
    lib/color_transform/ColorTransform.h \
    lib/color_transform/YCbCrColorTransform.h \
    lib/color_transform/EmptyColorTransform.h \
    lib/color_transform/ColorTransformFactory.h \
    lib/color_transform/YCbCrColorTransformCuda.h \
    lib/compressor/CudaWaveletCompressor.h \
    lib/util/HostEventTimer.h \
    lib/util/TimersData.h
SOURCES += lib/wavelet_transform/Wavelet.cpp \
    lib/wavelet_transform/WaveletFactory.cpp \
    test/transformTest.cpp \
    lib/util/WaveletCompressorUtil.cpp \
    lib/wavelet_transform/WaveletTransformImpl.cpp \
    lib/wavelet_transform/WaveletTransformCudaImpl.cpp \
    lib/wavelet_transform/CudaWaveletTransform.cxx \
    lib/wavelet_transform/WaveletTransformCudaImpl.cxx \
    lib/util/CudaEventTimer.cpp \
    lib/wavelet_transform/Filter.cpp \
    test/transformPerformanceTest.cpp \
    lib/quantizer/UniformQuantizer.cpp \
    test/quantizerTest.cpp \
    app/transformViewer.cpp \
    lib/common/OpenCvImage.cpp \
    lib/compressor/WaveletCompressor.cpp \
    lib/compressor/BitAllocator.cpp \
    lib/arithcoder/ArithCoder.cpp \
    lib/arithcoder/ArithCoderModel.cpp \
    lib/arithcoder/ArithCoderModelOrder0.cpp \
    test/arithCoderTest.cpp \
    app/waveletCompressor.cpp \
    lib/common/SubbandData.cpp \
    OTHER_FILES \
    += \
    SConstruct \
    build.properties \
    scons_tools/cuda.py \
    Doxyfile \
    scons_tools/doxygen.py \
    lib/common/OpenCvImageSequence.cpp \
    lib/quantizer/DeadzoneUniformQuantizer.cpp \
    lib/color_transform/YCbCrColorTransform.cpp \
    lib/color_transform/YCbCrColorTransformCuda.cxx \
    lib/color_transform/ColorTransformCuda.cxx \
    lib/compressor/CudaWaveletCompressor.cpp \
    lib/util/HostEventTimer.cpp \
    lib/util/TimersData.cpp
INCLUDEPATH += lib/
INCLUDEPATH += /home/stanislaw/NVIDIA_GPU_Computing_SDK/C/common/inc
INCLUDEPATH += /home/stanislaw/cuda_3.2/cuprintf

OTHER_FILES += \
    LICENSE.txt \
    README.txt
