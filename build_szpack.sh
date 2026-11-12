#!/usr/bin/env bash
# Build the SZpack v2 python extension (pymsz/szpack/SZpack.*.so) used by pymsz.SZpack_model.
#
# usage: ./build_szpack.sh [--build-gsl | --gsl-only]
#   --build-gsl   download GSL 2.8 and build a static copy in ./third_party/gsl
#                 (only needed if GSL is not installed, e.g. no `apt install libgsl-dev`)
#   --gsl-only    only build the static GSL (used to make the binary wheels, see pyproject.toml)
# environment:
#   PYTHON=/path/to/python   python used for the build (default: python3); needs pybind11 + setuptools
#   GSL_DIR=/prefix          use the GSL installed under this prefix
# Works on Linux and macOS.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
PYTHON=${PYTHON:-python3}

GSL_VERSION=2.8
GSL_SHA256=6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190

BUILD_GSL=0
BUILD_EXT=1
case "${1:-}" in
    --build-gsl) BUILD_GSL=1 ;;
    --gsl-only) BUILD_GSL=1; BUILD_EXT=0 ;;
    "") ;;
    *) echo "unknown option $1 (use --build-gsl or --gsl-only)" >&2; exit 1 ;;
esac

ncpu() { nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4; }

# 1. SZpack sources (git submodule pointing to https://github.com/CMBSPEC/SZpack)
if [ "${BUILD_EXT}" = 1 ] && [ ! -f SZpack/SZpack.cpp ]; then
    echo ">> fetching SZpack sources"
    if [ -f .gitmodules ] && grep -q SZpack .gitmodules; then
        git submodule update --init SZpack
    else
        git clone https://github.com/CMBSPEC/SZpack.git SZpack
    fi
fi

# 2. optional local GSL build
if [ "${BUILD_GSL}" = 1 ] && [ ! -f third_party/gsl/include/gsl/gsl_math.h ]; then
    echo ">> building GSL ${GSL_VERSION} in third_party/gsl"
    mkdir -p third_party/src
    cd third_party/src
    TARBALL=gsl-${GSL_VERSION}.tar.gz
    URL=https://ftp.gnu.org/gnu/gsl/${TARBALL}
    if [ ! -f "${TARBALL}" ]; then
        if command -v wget >/dev/null; then wget -q "${URL}"
        elif command -v curl >/dev/null; then curl -fsSLO "${URL}"
        else "${PYTHON}" -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "${URL}" "${TARBALL}"
        fi
    fi
    if command -v sha256sum >/dev/null; then
        echo "${GSL_SHA256}  ${TARBALL}" | sha256sum -c -
    else
        echo "${GSL_SHA256}  ${TARBALL}" | shasum -a 256 -c -
    fi
    rm -rf gsl-${GSL_VERSION} && tar xzf "${TARBALL}"
    cd gsl-${GSL_VERSION}
    ./configure --prefix="${ROOT}/third_party/gsl" --disable-shared --enable-static --with-pic CFLAGS="-O2 -fPIC"
    make -j"$(ncpu)"
    make install
    cd "${ROOT}"
fi
[ "${BUILD_EXT}" = 1 ] || exit 0

# 3. build the extension in place (pip install -e . / pip install . do the same)
"${PYTHON}" -c "import pybind11" 2>/dev/null || { echo "pybind11 missing: ${PYTHON} -m pip install pybind11"; exit 1; }
NPY_NUM_BUILD_JOBS=${NPY_NUM_BUILD_JOBS:-$(ncpu)} "${PYTHON}" setup.py build_ext --inplace

"${PYTHON}" -c "from pymsz.szpack import require; S = require(); print('SZpack extension OK:', S.__file__)"
