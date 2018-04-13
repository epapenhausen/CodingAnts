#!/bin/sh

keep=yes
verbose=yes

for option; do
	case "$option" in
		--keep)
			keep=yes
			;;
		--verbose)
			verbose=yes
			;;
	esac
done

EXEEXT=
DIR="/home/epapenhausen/CA/polybench-c-4.2.1-beta"
VERSION=ppcg-0.05
SIZE=-DLARGE_DATASET
#CC="gcc"
CC="nvcc"
HAVE_OPENCL=no
HAVE_CUDA=yes
HAVE_OPENMP=no
srcdir="."
if [ $keep = "yes" ]; then
	OUTDIR="out.$VERSION"
	mkdir "$OUTDIR" || exit 1
else
	if test "x$TMPDIR" = "x"; then
		TMPDIR=/tmp
	fi
	OUTDIR='/home/epapenhausen/CA/ppcg-0.05/qa'
	#OUTDIR=`mktemp -d $TMPDIR/ppcg.XXXXXXXXXX` || exit 1
fi
CPPFLAGS="-DPOLYBENCH_USE_C99 -DPOLYBENCH_DUMP_ARRAYS -DPOLYBENCH_TIME"
CPPFLAGS="$CPPFLAGS $SIZE -I $DIR/utilities"
#CFLAGS="-lm --std=gnu99"
CFLAGS="-lm -lstdc++"

echo "Running tests in folder ${OUTDIR}"

run_tests () {
	ext=$1

	ppcg_options=$2
	cc_options=$3

	if [ "x$ppcg_options" = "x" ]; then
		ppcg_option_str="none"
	else
		ppcg_option_str=$ppcg_options
	fi

	if [ "x$cc_options" = "x" ]; then
		cc_option_str="none"
	else
		cc_option_str=$cc_options
	fi

	echo Test: $ext, ppcg options: $ppcg_option_str, CC options: $cc_option_str
	for i in `cat $DIR/utilities/benchmark_list`; do
		echo $i
		name=`basename $i`
		name=${name%.c}
		#source_opt="${OUTDIR}/$name.$ext.c"
		source_opt="${OUTDIR}/$name"_"host.cu"
		device_opt="${OUTDIR}/$name"_"kernel.cu"
		prog_orig=${OUTDIR}/$name.orig${EXEEXT}
		prog_opt=${OUTDIR}/$name.$ext${EXEEXT}
		output_orig=${OUTDIR}/$name.orig.out
		output_opt=${OUTDIR}/$name.$ext.out
		dir=`dirname $i`
		if [ $verbose = "yes" ]; then
			echo ./ppcg$EXEEXT -I $DIR/$dir $DIR/$i \
				$CPPFLAGS -o $source_opt $ppcg_options
		fi
		# UNCOMMENT THIS
		./ppcg$EXEEXT -I $DIR/$dir $DIR/$i $CPPFLAGS \
			-o $source_opt $ppcg_options || exit
		gcc -I $DIR/$dir $CPPFLAGS $DIR/$i -o $prog_orig \
			$DIR/utilities/polybench.c $CFLAGS

		$prog_orig 2> $output_orig
		if [ $verbose = "yes" ]; then
			echo $CC -I $DIR/$dir $CPPFLAGS $source_opt \
				-o $prog_opt $DIR/utilities/polybench.c \
				$CFLAGS $cc_options
		fi
		$CC -I $DIR/utilities -I pet/include -I isl/include \
		    -I $DIR/$dir $DIR/utilities/polybench.c \
		    -arch sm_30 $CPPFLAGS $source_opt $device_opt -o $prog_opt \
		     $CFLAGS $cc_options || exit

		$prog_opt 2> $output_opt
		#cmp $output_orig $output_opt || exit
		python compare.py -i $output_orig -o $output_opt || exit
	done
}

#run_tests ppcg "--target=c --tile"
#run_tests ppcg_live "--target=c --no-live-range-reordering --tile"
#-
# Test OpenMP code, if compiler supports openmp
if [ $HAVE_OPENMP = "yes" ]; then
	run_tests ppcg_omp "--target=c --openmp" -fopenmp
	echo Introduced `grep -R 'omp parallel' "${OUTDIR}" | wc -l` '"pragma omp parallel for"'
else
	echo Compiler does not support OpenMP. Skipping OpenMP tests.
fi

if [ $HAVE_OPENCL = "yes" ]; then
	run_tests ppcg_opencl "--target=opencl --opencl-no-use-gpu" \
				"-I $srcdir $srcdir/ocl_utilities.c -lOpenCL"
fi

if [ $HAVE_CUDA = "yes" ]; then
    # --sizes={kernel[0]->block[64,64];}
    run_tests ppcg_cuda "--target=cuda --dump-sizes --coding-ants --sizes={kernel[0]->tile[32,32];kernel[0]->block[16];}" \
				"-I $srcdir -lcuda"
fi

if [ $keep = "no" ]; then
    rm -r "${OUTDIR}"
fi
