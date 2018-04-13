#!/bin/sh

DIR="/home/epapenhausen/CA/polybench-c-4.2.1-beta"
VERSION=ppcg-0.05
OUTDIR="out.$VERSION"
SIZE=-DLARGE_DATASET

CPPFLAGS="-DPOLYBENCH_USE_C99 -DPOLYBENCH_DUMP_ARRAYS -DPOLYBENCH_TIME -DDATA_TYPE_IS_FLOAT"
CPPFLAGS="$CPPFLAGS $SIZE -I $DIR/utilities"
CFLAGS="-lm -lstdc++"

CC="nvcc"

srcdir="."
dir=`dirname $1`
name=`basename $1`
name=${name%.c}
echo $dir
echo $name

source_opt="${OUTDIR}/$name"_"host.cu"
device_opt="${OUTDIR}/$name"_"kernel.cu"
prog_opt=${OUTDIR}/$name."ppcg_cuda"
cc_options="-I $srcdir -lcuda"

$CC -I $DIR/utilities -I pet/include -I isl/include \
    -I $dir $DIR/utilities/polybench.c \
    -arch sm_30 $CPPFLAGS $source_opt $device_opt -o $prog_opt \
    $CFLAGS $cc_options || exit
