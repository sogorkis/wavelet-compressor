wavelet-compressor - Wavelet compression tool
=============================================

Bsc thesis: "Wavelet compression in NVIDIA CUDA" Stanisław Ogórkis, 2011 - http://ogorkis.net/education/bsc/

1. Building and dependencies:
In order to work with code you will need following tools and libraries:
* scons
* cuda sdk
* opencv with highgui
* gtest
To build code you will probably need to set following variables:
* PATH
* LD_LIBRARY_PATH
* CPLUS_INCLUDE_PATH
* LIBRARY_PATH
Example:
```
export PATH="$PATH:/usr/local/cuda/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib"
export CPLUS_INCLUDE_PATH="/usr/local/cuda/include"
export LIBRARY_PATH="/usr/lib64/nvidia-current"
```
You can custimze your build by editing build.properties file. To build application just invoke following command:
```
scons
```
2. Compressor usage examples:
* help:
```
./waveletCompressor -help
Usage: waveletCompressor [OPTIONS] <input file name> <output file name>

OPTIONS:
-help              - prints this help
-verbose           - verbose output
-cuda              - uses cuda implementation
-wavelet <name>    - specify wavelet (haar, daub4, daub6, cdf97, antonini); compression only; default antonini
-ratio <value>     - specify compression ratio; compression only; required
-quantizer <name>  - specify quantizer type (utq, dutq); compression only; default dutq (deadzone uniform quant.)
-levels <value>    - specify number of decomposition levels; compression only; default 5
```
* image compression:
```
./waveletCompressor -ratio 16 -verbose test/resources/lena.tiff lena_encoded.wci
```
* image decompression:
```
./waveletCompressor -verbose lena_encoded.wci lena_decoded.tiff
```
* video sequence compression using cuda and specifying decomposition count:
```
./waveletCompressor -ratio 16 -verbose -cuda -levels 2 test/resources/crysis_micro.avi crysis_encoded.wcv
```
* video sequence decompression using cuda:
```
./waveletCompressor -verbose -cuda encoded.wcv crysis_decoded.avi
```
3. Transform Viewer application usage examples:
* compare two images using diff option:
```
./transformViewer -diff test/resources/lena.tiff lena_decoded.tiff
```
* compare two video sequences:
```
./transformViewer test/resources/crysis_micro.avi crysis_decoded.avi
```
