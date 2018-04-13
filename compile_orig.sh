#!/bin/sh

DIR="/home/epapenhausen/CA/polybench-c-4.2.1-beta"
VERSION=ppcg-0.05
OUTDIR="out.$VERSION"
SIZE=-DLARGE_DATASET

CPPFLAGS="-DPOLYBENCH_USE_C99 -DPOLYBENCH_DUMP_ARRAYS -DPOLYBENCH_TIME -DDATA_TYPE_IS_FLOAT"
CPPFLAGS="$CPPFLAGS $SIZE -I $DIR/utilities"
CFLAGS="-lm -lstdc++"

srcdir="."
dir=`dirname $1`
name=`basename $1`
name=${name%.c}

prog_orig=${OUTDIR}/$name.orig
cc_options="-I $srcdir -lcuda"

gcc -I $dir $CPPFLAGS $1 -o $prog_orig \
    $DIR/utilities/polybench.c $CFLAGS
