#!/bin/sh

DIR="/home/epapenhausen/CA/polybench-c-4.2.1-beta"
VERSION=ppcg-0.05
OUTDIR="out.$VERSION"
SIZE=-DLARGE_DATASET

CPPFLAGS="-DPOLYBENCH_USE_C99 -DPOLYBENCH_DUMP_ARRAYS -DPOLYBENCH_TIME -DDATA_TYPE_IS_FLOAT"
CPPFLAGS="$CPPFLAGS $SIZE -I $DIR/utilities"

dir=`dirname $1`
name=`basename $1`
name=${name%.c}
source_opt="${OUTDIR}/$name"_"host.cu"
ppcg_options="--target=cuda --dump-schedule --dump-sizes --coding-ants --ca-O 1 --save-schedule=$name"_"ref.sched" 
# --no-isl-schedule-maximize-band-depth"
# --load-schedule=$name.sched"


./ppcg$EXEEXT -I $DIR/$dir $DIR/$1 $CPPFLAGS \
    -o $source_opt $ppcg_options || exit
