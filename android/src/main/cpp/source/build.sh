#!/bin/bash
SDK_ROOT=/Users/wweng/Library/Android/sdk
NDK_ROOT=/Users/wweng/Library/Android/sdk/ndk/25.2.9519653
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
 
mkdir ${SCRIPT_DIR}/build
cd ${SCRIPT_DIR}/build
 
# please set SDK_ROOT, NDK_ROOT, etc. by env
# require download cmake, etc. by sdkmanager "cmake;3.18.1" "ndk;24.0.8215888" --channel=0 --sdk_root=./sdk_path
# require ndk >= 24.0.8215888
# for example:
# export SDK_ROOT=/root/codes/my_sdk/sdk/
# export NDK_ROOT=/root/codes/my_sdk/sdk/ndk/24.0.8215888/
 
if [ ! -d ${SDK_ROOT} ] ; then
    echo "ERROR: please set valid sdk path by env SDK_ROOT"
    exit 1
fi
if [ ! -d ${SDK_ROOT}/cmake/3.18.1/ ] ; then
    echo "ERROR: please download cmake 3.18.1 for sdk"
    exit 1
fi
if [ ! -d ${NDK_ROOT} ] ; then
    echo "ERROR: please set valid ndk path by env NDK_ROOT"
    exit 1
fi
if [ -z ${ANDROID_ABI} ] ; then
    ANDROID_ABI=arm64-v8a
fi
if [ -z ${MINSDKVERSION} ] ; then
    MINSDKVERSION=29
fi
 
echo "SDK_ROOT:" $SDK_ROOT
echo "NDK_ROOT:" $NDK_ROOT
echo "ANDROID_ABI:" $ANDROID_ABI
echo "MINSDKVERSION:" $MINSDKVERSION
 
${SDK_ROOT}/cmake/3.18.1/bin/cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=${NDK_ROOT}/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=${ANDROID_ABI} \
    -DANDROID_NDK=${NDK_ROOT} \
    -DANDROID_PLATFORM=android-${MINSDKVERSION} \
    -DCMAKE_ANDROID_ARCH_ABI=${ANDROID_ABI} \
    -DCMAKE_ANDROID_NDK=${NDK_ROOT} \
    -DCMAKE_MAKE_PROGRAM=${SDK_ROOT}/cmake/3.18.1/bin/ninja \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_SYSTEM_VERSION=${MINSDKVERSION} \
    -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBLA_VENDOR=OpenBLAS \
    -DMKL_LIBRARIES=/Users/wweng/Documents/projects/famifoto/OpenBLAS/build/lib/libopenblas.a \
    -DCMAKE_CXX_FLAGS_RELEASE="-s"  \
    -GNinja \
    ..
if [ $? -ne 0 ]; then
    echo "ERROR: cmake failed"
    exit 1
fi
 
${SDK_ROOT}/cmake/3.18.1/bin/ninja
if [ $? -ne 0 ]; then
    echo "ERROR: build failed"
    exit 1
fi
 