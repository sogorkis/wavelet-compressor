import ConfigParser

cfg = ConfigParser.ConfigParser()
cfg.read("build.properties")
DEBUG		= cfg.getboolean("build", "DEBUG")
CUDA		= cfg.getboolean("build", "CUDA")
AUTO_TEST	= cfg.getboolean("build", "AUTO_TEST")

env = Environment()

env.Append( LIBS=['cv', 'highgui', 'rt', 'wavelet'])
env.Append( LIBPATH='.' )
env.Append( CPPPATH=['lib'] )
env.Append( CPPFLAGS=['-Werror', '-Wall', '-Wextra', '-Wno-unused'] )

if ARGUMENTS.get('doc', 0):
	env.Tool('doxygen', toolpath=['scons_tools'])
	env.Doxygen("Doxyfile")

if CUDA:
	env.Tool('cuda', toolpath=['scons_tools'])
	env.Append( LIBS=['cutil', 'cuda', 'cudart'])
        env.Append( NVCCFLAGS='-arch=sm_11' )

if DEBUG:
        env.Append( CPPFLAGS=['-ggdb'] )
else:
        env.Append( CPPFLAGS=['-O2'] )

# lib files
lib_files = Split("""
lib/arithcoder/ArithCoder.cpp
lib/arithcoder/ArithCoderModel.cpp
lib/arithcoder/ArithCoderModelOrder0.cpp

lib/color_transform/YCbCrColorTransform.cpp

lib/common/OpenCvImage.cpp
lib/common/OpenCvImageSequence.cpp
lib/common/SubbandData.cpp

lib/compressor/BitAllocator.cpp
lib/compressor/WaveletCompressor.cpp

lib/quantizer/DeadzoneUniformQuantizer.cpp
lib/quantizer/UniformQuantizer.cpp

lib/util/WaveletCompressorUtil.cpp
lib/util/HostEventTimer.cpp
lib/util/TimersData.cpp

lib/wavelet_transform/Filter.cpp
lib/wavelet_transform/Wavelet.cpp
lib/wavelet_transform/WaveletFactory.cpp
lib/wavelet_transform/WaveletTransformImpl.cpp
""")

# lib files for cuda implementation
if CUDA:
	lib_files += Split("""
	lib/util/CudaEventTimer.cpp

        lib/color_transform/YCbCrColorTransformCuda.cu

        lib/compressor/CudaWaveletCompressor.cpp

	lib/wavelet_transform/WaveletTransformCudaImpl.cu
	""")

env.StaticLibrary('wavelet', lib_files)

env.Program('transformViewer', ['app/transformViewer.cpp'] )
env.Program('waveletCompressor', ['app/waveletCompressor.cpp'] )


# automatic unit testing
def builder_unit_test(target, source, env):
	import os
	app = str(source[0].abspath)
	if os.spawnl(os.P_WAIT, app, app)==0:
       		open(str(target[0]),'w').write("PASSED\n")
	else:
       		return 1

if AUTO_TEST:
	test_env = env.Clone()
	test_env.Append( LIBS=['wavelet', 'gtest_main'] )

	# Create a builder for tests
	bld = Builder(action = builder_unit_test)

	testQuantizer = test_env.Program('test/quantizer.test', ['test/quantizerTest.cpp'] )
	testArithCoder = test_env.Program('test/arithcoder.test', ['test/arithCoderTest.cpp'] )

        test_env.Append(BUILDERS = {'Test' :  bld})

	if CUDA:
		test_env.Program('test/transform.performanceTest', ['test/transformPerformanceTest.cpp'] )
		testTransform = test_env.Program('test/transform.test', ['test/transformTest.cpp'] )
                #test_env.Test("test.transform.passed", testTransform)

        #test_env.Test("test.quantizer.passed", testQuantizer)
        test_env.Test("test.arithCoder.passed", testArithCoder)
