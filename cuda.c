/*
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */
#include <assert.h>
#include <isl/aff.h>
#include <isl/ast.h>

#include "cuda_common.h"
#include "cuda.h"
#include "gpu.h"
#include "gpu_print.h"
#include "print.h"
#include "util.h"
#include "ca.h"

int kernel_time_only = 0;
int event_destroyed = 0;
static __isl_give isl_printer *print_cuda_macros(__isl_take isl_printer *p)
{
	const char *macros =
		"#define cudaCheckReturn(ret) \\\n"
		"  do { \\\n"
		"    cudaError_t cudaCheckReturn_e = (ret); \\\n"
		"    if (cudaCheckReturn_e != cudaSuccess) { \\\n"
		"      fprintf(stderr, \"CUDA error: %s\\n\", "
		"cudaGetErrorString(cudaCheckReturn_e)); \\\n"
		"      fflush(stderr); \\\n"
		"    } \\\n"
		"    assert(cudaCheckReturn_e == cudaSuccess); \\\n"
		"  } while(0)\n"
		"#define cudaCheckKernel() \\\n"
		"  do { \\\n"
		"    cudaCheckReturn(cudaGetLastError()); \\\n"
		"  } while(0)\n\n";

	p = isl_printer_print_str(p, macros);
	return p;
}

/* Print a declaration for the device array corresponding to "array" on "p".
 */
static __isl_give isl_printer *declare_device_array(__isl_take isl_printer *p,
	struct gpu_array_info *array)
{
	int i;

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, array->type);
	p = isl_printer_print_str(p, " ");
	if (!array->linearize && array->n_index > 1)
		p = isl_printer_print_str(p, "(");
	p = isl_printer_print_str(p, "*dev_");
	p = isl_printer_print_str(p, array->name);
	if (!array->linearize && array->n_index > 1) {
		p = isl_printer_print_str(p, ")");
		for (i = 1; i < array->n_index; i++) {
			p = isl_printer_print_str(p, "[");
			p = isl_printer_print_pw_aff(p, array->bound[i]);
			p = isl_printer_print_str(p, "]");
		}
	}
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	return p;
}

static __isl_give isl_printer *declare_texture_objects(__isl_take isl_printer *p,
	struct gpu_array_info *array)
{
        // declare the format desriptor
  	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaChannelFormatDesc");
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p, "channelDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	// declare the cudaExtent
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "struct cudaExtent");
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p, "cudaExtent_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	// declare the cudaArray
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaArray");
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p, "*cuArray_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	// declare the resource desriptor
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "struct cudaResourceDesc");
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p, "resDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	// declare the texture desriptor
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "struct cudaTextureDesc");
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p, "texDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	// declare the texture object
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaTextureObject_t");
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p, "tex_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, " = 0;");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

	return p;
}
						       
static __isl_give isl_printer *declare_device_arrays(__isl_take isl_printer *p,
	struct gpu_prog *prog)
{
	int i;

	for (i = 0; i < prog->n_array; ++i) {
	  if (!gpu_array_requires_device_allocation(&prog->array[i])) {
	    continue;
	  }
	  
	  if (prog->array[i].texture) {
	    // declare texture memory
	    p = declare_texture_objects(p, &prog->array[i]);
	  } 
	  else {
	    p = declare_device_array(p, &prog->array[i]);
	  }
	}
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);
	return p;
}

static __isl_give isl_printer *allocate_texture_objects(__isl_take isl_printer *p,
	struct gpu_array_info *array)
{
        assert(array->n_index >= 1 || array->n_index <= 3);
        int i;

	// should be either a float or an int
        assert(!strcmp(array->type, "float") || !strcmp(array->type, "int"));
        
	int float_type = (strcmp(array->type, "float") == 0);
	// allocate the channel descriptor
        p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "channelDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, " = ");
	p = isl_printer_print_str(p, "cudaCreateChannelDesc(");
	p = isl_printer_print_str(p, "8 * sizeof("); // needs to be num bits
	p = isl_printer_print_str(p, array->type);
	p = isl_printer_print_str(p, "), ");

	// do not handle float4, short2, etc. for now
	p = isl_printer_print_str(p, "0, ");
	p = isl_printer_print_str(p, "0, ");
	p = isl_printer_print_str(p, "0, ");
	
	if (float_type) {
	  p = isl_printer_print_str(p, "cudaChannelFormatKindFloat");
	}
	else {
	  p = isl_printer_print_str(p, "cudaChannelFormatKindSigned");
	}
	p = isl_printer_print_str(p, ");");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

	// allocate the cudaExtent
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaExtent_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, " = ");
	p = isl_printer_print_str(p, "make_cudaExtent(");
	for (i = 0; i < array->n_index; ++i) {
	  p = isl_printer_print_pw_aff(p, array->bound[array->n_index - i - 1]);
	  if (i < 2) {
	    p = isl_printer_print_str(p, ", ");
	  }
	}
	
	for (i ; i < 3; ++i) {
	  p = isl_printer_print_str(p, "0");	    
	  if (i < 2) {
	    p = isl_printer_print_str(p, ", ");	  
	  }
	}
	p = isl_printer_print_str(p, ");");
	p = isl_printer_end_line(p);
	
	// allocate the cuda array
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaCheckReturn(cudaMalloc3DArray(&cuArray_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");
	p = isl_printer_print_str(p, "&channelDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");
	p = isl_printer_print_str(p, "cudaExtent_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ")); ");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

	return p;	
}

static __isl_give isl_printer *allocate_device_arrays(
	__isl_take isl_printer *p, struct gpu_prog *prog)
{
	int i;

	for (i = 0; i < prog->n_array; ++i) {
	  if (!gpu_array_requires_device_allocation(&prog->array[i])) {
	    continue;
	  }

	  if (prog->array[i].texture) {
	    // declare texture memory
	    p = allocate_texture_objects(p, &prog->array[i]);
	  } 
	  else {
	    p = isl_printer_start_line(p);
	    p = isl_printer_print_str(p,
				      "cudaCheckReturn(cudaMalloc((void **) &dev_");
	    p = isl_printer_print_str(p, prog->array[i].name);
	    p = isl_printer_print_str(p, ", ");
	    p = gpu_array_info_print_size(p, &prog->array[i]);
	    p = isl_printer_print_str(p, "));");
	    p = isl_printer_end_line(p);
	  }
	}
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);
	return p;
}

static __isl_give isl_printer *copy_texture_to_device(__isl_take isl_printer *p,
	struct gpu_array_info *array)
{
        int i;
        p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaCheckReturn(cudaMemcpyToArray(cuArray_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", 0, 0, ");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");
	p = gpu_array_info_print_size(p, array);
	p = isl_printer_print_str(p, ", cudaMemcpyHostToDevice));");
	p = isl_printer_end_line(p);

	// allocate the resource descriptor
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "memset(&resDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", 0, sizeof(");
	p = isl_printer_print_str(p, "resDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, "));");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "resDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ".resType = ");
	p = isl_printer_print_str(p, "cudaResourceTypeArray;");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "resDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ".res.array.array = ");
	p = isl_printer_print_str(p, "cuArray_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

	// allocate the texture descriptor
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "memset(&texDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", 0, sizeof(");
	p = isl_printer_print_str(p, "texDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, "));");
	p = isl_printer_end_line(p);
	
	for (i = 0; i < array->n_index; ++i) {
	  char buf[5];
	  snprintf(buf, 5, "%d", i);

	  p = isl_printer_start_line(p);
	  p = isl_printer_print_str(p, "texDesc_");
	  p = isl_printer_print_str(p, array->name);
	  p = isl_printer_print_str(p, ".addressMode[");
	  p = isl_printer_print_str(p, buf);
	  p = isl_printer_print_str(p, "] = ");
	  p = isl_printer_print_str(p, "cudaAddressModeClamp;");
	  p = isl_printer_end_line(p);
	}

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "texDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ".filterMode = ");
	p = isl_printer_print_str(p, "cudaFilterModePoint;");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "texDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ".readMode = ");
	p = isl_printer_print_str(p, "cudaReadModeElementType;");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "texDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ".normalizedCoords = ");
	p = isl_printer_print_str(p, "0;");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

	// allocate the cuda texture
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaCheckReturn(cudaCreateTextureObject(&tex_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");
	p = isl_printer_print_str(p, "&resDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");
	p = isl_printer_print_str(p, "&texDesc_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", NULL");
	p = isl_printer_print_str(p, ")); ");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);
	return p;
}

/* Print code to "p" for copying "array" from the host to the device
 * in its entirety.  The bounds on the extent of "array" have
 * been precomputed in extract_array_info and are used in
 * gpu_array_info_print_size.
 */
static __isl_give isl_printer *copy_array_to_device(__isl_take isl_printer *p,
	struct gpu_array_info *array)
{
        if (array->texture) {
	  return copy_texture_to_device(p, array);
        }

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaCheckReturn(cudaMemcpy(dev_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");

	if (gpu_array_is_scalar(array))
		p = isl_printer_print_str(p, "&");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");

	p = gpu_array_info_print_size(p, array);
	p = isl_printer_print_str(p, ", cudaMemcpyHostToDevice));");
	p = isl_printer_end_line(p);

	return p;
}

/* Print code to "p" for copying "array" back from the device to the host
 * in its entirety.  The bounds on the extent of "array" have
 * been precomputed in extract_array_info and are used in
 * gpu_array_info_print_size.
 */
static __isl_give isl_printer *copy_array_from_device(
	__isl_take isl_printer *p, struct gpu_array_info *array)
{

	if (kernel_time_only && !event_destroyed) {
	  //cudaCheckReturn(cudaEventRecord(stop, 0));
	  p = isl_printer_start_line(p);
	  p = isl_printer_print_str(p, "cudaCheckReturn(cudaEventRecord(stop, 0));");
	  p = isl_printer_end_line(p);

	  //cudaCheckReturn(cudaEventSynchronize (stop) );

	  p = isl_printer_start_line(p);
	  p = isl_printer_print_str(p, "cudaCheckReturn(cudaEventSynchronize (stop) );");
	  p = isl_printer_end_line(p);

	  //  cudaCheckReturn(cudaEventElapsedTime(&elapsed, start, stop));
	  p = isl_printer_start_line(p);
	  p = isl_printer_print_str(p, "cudaCheckReturn(cudaEventElapsedTime(&elapsed, start, stop));");
	  p = isl_printer_end_line(p);

	 // cudaCheckReturn(cudaEventDestroy(start));
	  p = isl_printer_start_line(p);
	  p = isl_printer_print_str(p, "cudaCheckReturn(cudaEventDestroy(start));");
	  p = isl_printer_end_line(p);

	  //  cudaCheckReturn(cudaEventDestroy(stop));
	  p = isl_printer_start_line(p);
	  p = isl_printer_print_str(p, "cudaCheckReturn(cudaEventDestroy(stop));");
	  p = isl_printer_end_line(p);

	  p = isl_printer_start_line(p);
	  p = isl_printer_print_str(p, "printf(\"%f\\n\", (elapsed / 1000.0f));");
	  p = isl_printer_end_line(p);
	  event_destroyed = 1;
	}

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaCheckReturn(cudaMemcpy(");
	if (gpu_array_is_scalar(array))
		p = isl_printer_print_str(p, "&");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", dev_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");
	p = gpu_array_info_print_size(p, array);
	p = isl_printer_print_str(p, ", cudaMemcpyDeviceToHost));");
	p = isl_printer_end_line(p);

	return p;
}

static void print_reverse_list(FILE *out, int len, int *list)
{
	int i;

	if (len == 0)
		return;

	fprintf(out, "(");
	for (i = 0; i < len; ++i) {
		if (i)
			fprintf(out, ", ");
		fprintf(out, "%d", list[len - 1 - i]);
	}
	fprintf(out, ")");
}

/* Print the effective grid size as a list of the sizes in each
 * dimension, from innermost to outermost.
 */
static __isl_give isl_printer *print_grid_size(__isl_take isl_printer *p,
	struct ppcg_kernel *kernel)
{
	int i;
	int dim;

	dim = isl_multi_pw_aff_dim(kernel->grid_size, isl_dim_set);
	if (dim == 0)
		return p;

	p = isl_printer_print_str(p, "(");
	for (i = dim - 1; i >= 0; --i) {
		isl_pw_aff *bound;

		bound = isl_multi_pw_aff_get_pw_aff(kernel->grid_size, i);
		p = isl_printer_print_pw_aff(p, bound);
		isl_pw_aff_free(bound);

		if (i > 0)
			p = isl_printer_print_str(p, ", ");
	}

	p = isl_printer_print_str(p, ")");

	return p;
}

/* Print the grid definition.
 */
static __isl_give isl_printer *print_grid(__isl_take isl_printer *p,
	struct ppcg_kernel *kernel)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "dim3 k");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "_dimGrid");
	p = print_grid_size(p, kernel);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	return p;
}

/* Print the arguments to a kernel declaration or call.  If "types" is set,
 * then print a declaration (including the types of the arguments).
 *
 * The arguments are printed in the following order
 * - the arrays accessed by the kernel
 * - the parameters
 * - the host loop iterators
 */
static __isl_give isl_printer *print_kernel_arguments(__isl_take isl_printer *p,
	struct gpu_prog *prog, struct ppcg_kernel *kernel, int types)
{
	int i, n;
	int first = 1;
	unsigned nparam;
	isl_space *space;
	const char *type;
	printf("kernel %d\n", kernel->id);
	for (i = 0; i < prog->n_array; ++i) {
		int required;

		required = ppcg_kernel_requires_array_argument(kernel, i);		
		if (required < 0)
			return isl_printer_free(p);

		if (!required)
			continue;

		if (!first)
			p = isl_printer_print_str(p, ", ");

		if (types)
			p = gpu_array_info_print_declaration_argument(p,
				&prog->array[i], NULL);
		else
			p = gpu_array_info_print_call_argument(p,
				&prog->array[i]);

		first = 0;
	}

	space = isl_union_set_get_space(kernel->arrays);
	nparam = isl_space_dim(space, isl_dim_param);
	for (i = 0; i < nparam; ++i) {
		const char *name;

		name = isl_space_get_dim_name(space, isl_dim_param, i);

		if (!first)
			p = isl_printer_print_str(p, ", ");
		if (types)
			p = isl_printer_print_str(p, "int ");
		p = isl_printer_print_str(p, name);

		first = 0;
	}
	isl_space_free(space);

	n = isl_space_dim(kernel->space, isl_dim_set);
	type = isl_options_get_ast_iterator_type(prog->ctx);
	for (i = 0; i < n; ++i) {
		const char *name;

		if (!first)
			p = isl_printer_print_str(p, ", ");
		name = isl_space_get_dim_name(kernel->space, isl_dim_set, i);
		if (types) {
			p = isl_printer_print_str(p, type);
			p = isl_printer_print_str(p, " ");
		}
		p = isl_printer_print_str(p, name);

		first = 0;
	}
	printf("end\n");
	return p;
}

/* Print the header of the given kernel.
 */
static __isl_give isl_printer *print_kernel_header(__isl_take isl_printer *p,
	struct gpu_prog *prog, struct ppcg_kernel *kernel)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "__global__ void kernel");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "(");
	p = print_kernel_arguments(p, prog, kernel, 1);
	p = isl_printer_print_str(p, ")");

	return p;
}

/* Print the header of the given kernel to both gen->cuda.kernel_h
 * and gen->cuda.kernel_c.
 */
static void print_kernel_headers(struct gpu_prog *prog,
	struct ppcg_kernel *kernel, struct cuda_info *cuda)
{
	isl_printer *p;

	p = isl_printer_to_file(prog->ctx, cuda->kernel_h);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = print_kernel_header(p, prog, kernel);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);
	isl_printer_free(p);

	p = isl_printer_to_file(prog->ctx, cuda->kernel_c);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = print_kernel_header(p, prog, kernel);
	p = isl_printer_end_line(p);
	isl_printer_free(p);
}

/* Print the header of the given kernel.
 */
static __isl_give isl_printer *print_warp_reduce(__isl_take isl_printer *p)
{
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "__inline__ __device__ float warpReduce(float val) {");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, " for (int offset = (blockDim.x > 32 ? 32 : blockDim.x) >> 1; offset > 0; offset >>= 1) ");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "   val += __shfl_down(val, offset); ");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, " return val; ");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "}");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);
	
	return p;
}

static void print_reduce(__isl_take isl_printer *p)
{
  p = isl_printer_start_line(p);
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "#define reduce(w, r) { \\");
  p = isl_printer_end_line(p);
  
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "  float red = warpReduce(r); \\");
  p = isl_printer_end_line(p);
  
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "  unsigned int laneid; \\");
  p = isl_printer_end_line(p); 

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "  asm(\"mov.u32 %0, %%laneid;\" : \"=r\"(laneid)); \\");
  p = isl_printer_end_line(p); 

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "  if (laneid == 0) \\");
  p = isl_printer_end_line(p);
 
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "    atomicAdd(&w, red); \\");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "}");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_end_line(p);
}

static void print_kernel_macros(struct gpu_prog *prog,
				struct cuda_info *cuda)
{
  isl_printer *p;
  p = isl_printer_to_file(prog->ctx, cuda->kernel_c);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  print_warp_reduce(p);
  print_reduce(p);
}
static void print_indent(FILE *dst, int indent)
{
	fprintf(dst, "%*s", indent, "");
}

/* Print a list of iterators of type "type" with names "ids" to "out".
 * Each iterator is assigned one of the cuda identifiers in cuda_dims.
 * In particular, the last iterator is assigned the x identifier
 * (the first in the list of cuda identifiers).
 */
static void print_iterators(FILE *out, const char *type,
	__isl_keep isl_id_list *ids, const char *cuda_dims[])
{
	int i, n;

	n = isl_id_list_n_id(ids);
	if (n <= 0)
		return;
	print_indent(out, 4);
	fprintf(out, "%s ", type);
	for (i = 0; i < n; ++i) {
		isl_id *id;

		if (i)
			fprintf(out, ", ");
		id = isl_id_list_get_id(ids, i);
		fprintf(out, "%s = %s", isl_id_get_name(id),
			cuda_dims[n - 1 - i]);
		isl_id_free(id);
	}
	fprintf(out, ";\n");
}

static void print_kernel_iterators(FILE *out, struct ppcg_kernel *kernel)
{
	isl_ctx *ctx = isl_ast_node_get_ctx(kernel->tree);
	const char *type;
	const char *block_dims[] = { "blockIdx.x", "blockIdx.y" };
	const char *thread_dims[] = { "threadIdx.x", "threadIdx.y",
					"threadIdx.z" };

	type = isl_options_get_ast_iterator_type(ctx);

	print_iterators(out, type, kernel->block_ids, block_dims);
	print_iterators(out, type, kernel->thread_ids, thread_dims);
}

static __isl_give isl_printer *print_kernel_var(__isl_take isl_printer *p,
	struct ppcg_kernel_var *var)
{
	int j;

	p = isl_printer_start_line(p);
	if (var->type == ppcg_access_shared)
		p = isl_printer_print_str(p, "__shared__ ");
	p = isl_printer_print_str(p, var->array->type);
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p,  var->name);
	for (j = 0; j < var->array->n_index; ++j) {
		isl_val *v;

		p = isl_printer_print_str(p, "[");
		v = isl_vec_get_element_val(var->size, j);

		if (j > 0 && j == var->array->n_index - 1) {
		  // coarse handling of shared memory bank conflicts
		  isl_ctx *ctx = isl_val_get_ctx(v);
		  long vsi = isl_val_get_num_si(v);
		  isl_val_free(v);
		  v = isl_val_int_from_si(ctx, vsi + 1);
		}

		p = isl_printer_print_val(p, v);
		isl_val_free(v);
		p = isl_printer_print_str(p, "]");
	}
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	return p;
}

static __isl_give isl_printer *print_kernel_vars(__isl_take isl_printer *p,
	struct ppcg_kernel *kernel)
{
	int i;

	for (i = 0; i < kernel->n_var; ++i)
		p = print_kernel_var(p, &kernel->var[i]);

	return p;
}

/* Print a sync statement.
 */
static __isl_give isl_printer *print_sync(__isl_take isl_printer *p,
	struct ppcg_kernel_stmt *stmt)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "__syncthreads();");
	p = isl_printer_end_line(p);

	return p;
}

/* This function is called for each user statement in the AST,
 * i.e., for each kernel body statement, copy statement or sync statement.
 */
static __isl_give isl_printer *print_kernel_stmt(__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options,
	__isl_keep isl_ast_node *node, void *user)
{
	isl_id *id;
	struct ppcg_kernel_stmt *stmt;

	id = isl_ast_node_get_annotation(node);
	stmt = isl_id_get_user(id);
	isl_id_free(id);
	isl_ast_print_options_free(print_options);

	switch (stmt->type) {
	case ppcg_kernel_copy:
		return ppcg_kernel_print_copy(p, stmt);
	case ppcg_kernel_sync:
		return print_sync(p, stmt);
	case ppcg_kernel_domain:
		return ppcg_kernel_print_domain(p, stmt);
	}

	return p;
}

static isl_stat get_stmt_str_array(__isl_take isl_set *set, void *user) {
  struct stmt_arr *arr = (struct stmt_arr *)user;
  isl_space *space = isl_set_get_space(set);
  char *space_str = strdup(isl_space_to_str(space));

  char *name_start = strstr(space_str, "S_");
  char *name_end = strstr(name_start, "[");
  size_t name_size = (name_end - name_start);

  char *sname = (char *)calloc(name_size + 1, sizeof(char));
  strncpy(sname, name_start, name_size);

  arr->stmt_str[arr->len++] = strdup(sname);
  return isl_stat_ok;
}

static void print_kernel(struct gpu_prog *prog, struct ppcg_kernel *kernel,
	struct cuda_info *cuda)
{
	isl_ctx *ctx = isl_ast_node_get_ctx(kernel->tree);
	isl_ast_print_options *print_options;
	isl_printer *p;
	
	print_kernel_headers(prog, kernel, cuda);
	fprintf(cuda->kernel_c, "{\n");
	print_kernel_iterators(cuda->kernel_c, kernel);

	p = isl_printer_to_file(ctx, cuda->kernel_c);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = isl_printer_indent(p, 4);

	p = print_kernel_vars(p, kernel);
	p = isl_printer_end_line(p);
	p = isl_ast_op_type_print_macro(isl_ast_op_fdiv_q, p);
	p = ppcg_print_macros(p, kernel->tree);
      
	int unroll = 0;
	if (cuda->wrap) {
	  struct ca_wrapper *wrap = cuda->wrap;
	  int n_stmts = isl_union_set_n_set(kernel->core);
	  char **stmts = (char **)calloc(n_stmts, sizeof(char*));

	  struct stmt_arr arr;
	  arr.stmt_str = stmts;
	  arr.len = 0;

	  isl_union_set_foreach_set(kernel->core, &get_stmt_str_array, &arr);
	  
	  unroll = 0;//wrap->traverse_unroll(wrap->ant, wrap->s_path, wrap->node, stmts, n_stmts);
	  
	  // update wrap node to point to the current leaf
	  struct ca_node *node = wrap->node;
	  struct path *s_path = wrap->s_path;
	  while(s_path->next != NULL) {
	    assert(node != NULL);
	    assert(node->out_edges != NULL);
	    assert(node->n_edges > 0);
    
	    assert(s_path->v < node->n_edges);
	    node = node->out_edges[s_path->v].dst;
	    node->kernel_id = kernel->id;
	    node->in_kernel = 1;

	    s_path = s_path->next;
    
	    assert(s_path != NULL);
	  }
	  assert(node != NULL);

	  wrap->s_path = s_path;
	  wrap->node = node;

	  printf("kernel %d, unroll %d\n", kernel->id, unroll);
	}

	print_options = isl_ast_print_options_alloc_unroll(ctx, unroll);
	print_options = isl_ast_print_options_set_print_user(print_options,
							     &print_kernel_stmt, NULL);
	
	p = isl_ast_node_print(kernel->tree, p, print_options);
	isl_printer_free(p);

	fprintf(cuda->kernel_c, "}\n");
}

/* Print a statement for copying an array to or from the device.
 * The statement identifier is called "to_device_<array name>" or
 * "from_device_<array name>" and its user pointer points
 * to the gpu_array_info of the array that needs to be copied.
 *
 * Extract the array from the identifier and call
 * copy_array_to_device or copy_array_from_device.
 */
static __isl_give isl_printer *print_to_from_device(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node, struct gpu_prog *prog)
{
	isl_ast_expr *expr, *arg;
	isl_id *id;
	const char *name;
	struct gpu_array_info *array;

	expr = isl_ast_node_user_get_expr(node);
	arg = isl_ast_expr_get_op_arg(expr, 0);
	id = isl_ast_expr_get_id(arg);
	name = isl_id_get_name(id);
	array = isl_id_get_user(id);
	isl_id_free(id);
	isl_ast_expr_free(arg);
	isl_ast_expr_free(expr);

	if (!name)
		array = NULL;
	if (!array)
		return isl_printer_free(p);

	if (!prefixcmp(name, "to_device"))
		return copy_array_to_device(p, array);
	else
		return copy_array_from_device(p, array);
}

struct print_host_user_data {
	struct cuda_info *cuda;
	struct gpu_prog *prog;
};



/* Print the user statement of the host code to "p".
 *
 * The host code may contain original user statements, kernel launches and
 * statements that copy data to/from the device.
 * The original user statements and the kernel launches have
 * an associated annotation, while the data copy statements do not.
 * The latter are handled by print_to_from_device.
 * The annotation on the user statements is called "user".
 *
 * In case of a kernel launch, print a block of statements that
 * defines the grid and the block and then launches the kernel.
 */
static __isl_give isl_printer *print_host_user(__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options,
	__isl_keep isl_ast_node *node, void *user)
{
	isl_id *id;
	int is_user;
	struct ppcg_kernel *kernel;
	struct ppcg_kernel_stmt *stmt;
	struct print_host_user_data *data;

	isl_ast_print_options_free(print_options);

	data = (struct print_host_user_data *) user;

	id = isl_ast_node_get_annotation(node);
	if (!id) {
		return print_to_from_device(p, node, data->prog);
	}

	is_user = !strcmp(isl_id_get_name(id), "user");
	kernel = is_user ? NULL : isl_id_get_user(id);
	stmt = is_user ? isl_id_get_user(id) : NULL;
	isl_id_free(id);

	if (is_user) {
		return ppcg_kernel_print_domain(p, stmt);
	}

	if (kernel->id == 0) {

	  if (kernel_time_only) {
	    p = isl_printer_start_line(p);
	    p = isl_printer_print_str(p, "cudaCheckReturn(cudaEventCreate(&start));");
	    p = isl_printer_end_line(p);

	    p = isl_printer_start_line(p);
	    p = isl_printer_print_str(p, "cudaCheckReturn(cudaEventCreate(&stop));");
	    p = isl_printer_end_line(p);

	    p = isl_printer_start_line(p);
	    p = isl_printer_print_str(p, "cudaCheckReturn( cudaEventRecord(start, 0));");
	    p = isl_printer_end_line(p);
	  }

	}

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "{");
	p = isl_printer_end_line(p);
	p = isl_printer_indent(p, 2);

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "dim3 k");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "_dimBlock");
	print_reverse_list(isl_printer_get_file(p),
				kernel->n_block, kernel->block_dim);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	p = print_grid(p, kernel);

	if (data->cuda->wrap) {  
	  int cache_pref = kernel->options->cuda_cache_config;

	  p = isl_printer_start_line(p);
	  p = isl_printer_print_str(p, "cudaCheckReturn(cudaFuncSetCacheConfig(");
	  p = isl_printer_print_str(p, "kernel");
	  p = isl_printer_print_int(p, kernel->id);
	  p = isl_printer_print_str(p, ", ");
	  if (cache_pref == 0)
	    p = isl_printer_print_str(p, "cudaFuncCachePreferShared));");
	  else if(cache_pref == 1)
	    p = isl_printer_print_str(p, "cudaFuncCachePreferNone));");
	  else if(cache_pref == 2)
	    p = isl_printer_print_str(p, "cudaFuncCachePreferL1));");

	    p = isl_printer_end_line(p);

	} // end if wrap

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "kernel");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, " <<<k");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "_dimGrid, k");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "_dimBlock>>> (");
	p = print_kernel_arguments(p, data->prog, kernel, 0);
	p = isl_printer_print_str(p, ");");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaCheckKernel();");
	p = isl_printer_end_line(p);

	p = isl_printer_indent(p, -2);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "}");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

	print_kernel(data->prog, kernel, data->cuda);

	return p;
}

static __isl_give isl_printer *print_host_code(__isl_take isl_printer *p,
	struct gpu_prog *prog, __isl_keep isl_ast_node *tree,
	struct cuda_info *cuda)
{
	isl_ast_print_options *print_options;
	isl_ctx *ctx = isl_ast_node_get_ctx(tree);
	struct print_host_user_data data = { cuda, prog };


	print_options = isl_ast_print_options_alloc(ctx);
	print_kernel_macros(prog, cuda);

	print_options = isl_ast_print_options_set_print_user(print_options,
						&print_host_user, &data);

	if (kernel_time_only) {
	  event_destroyed = 0;
	  p = isl_printer_start_line(p);
	  p = isl_printer_print_str(p, "float elapsed=0;");
	  p = isl_printer_end_line(p);

	  p = isl_printer_start_line(p);
	  p = isl_printer_print_str(p, "cudaEvent_t start, stop;");
	  p = isl_printer_end_line(p);
	}

	p = ppcg_print_macros(p, tree);
	p = isl_ast_node_print(tree, p, print_options);

	return p;
}

static __isl_give isl_printer *free_device_arrays(__isl_take isl_printer *p,
	struct gpu_prog *prog)
{
	int i;

	for (i = 0; i < prog->n_array; ++i) {
		if (!gpu_array_requires_device_allocation(&prog->array[i]))
			continue;

		if (prog->array[i].texture) {
		  p = isl_printer_start_line(p);
		  p = isl_printer_print_str(p, 
					    "cudaCheckReturn(cudaDestroyTextureObject(tex_");
		  p = isl_printer_print_str(p, prog->array[i].name);
		  p = isl_printer_print_str(p, "));");
		  p = isl_printer_end_line(p);

		  p = isl_printer_start_line(p);
		  p = isl_printer_print_str(p, "cudaCheckReturn(cudaFreeArray(cuArray_");
		  p = isl_printer_print_str(p, prog->array[i].name);
		  p = isl_printer_print_str(p, "));");
		  p = isl_printer_end_line(p);
		}
		else {
		  p = isl_printer_start_line(p);
		  p = isl_printer_print_str(p, "cudaCheckReturn(cudaFree(dev_");
		  p = isl_printer_print_str(p, prog->array[i].name);
		  p = isl_printer_print_str(p, "));");
		  p = isl_printer_end_line(p);
		}
	}

	return p;
}

/* Given a gpu_prog "prog" and the corresponding transformed AST
 * "tree", print the entire CUDA code to "p".
 * "types" collects the types for which a definition has already
 * been printed.
 */
static __isl_give isl_printer *print_cuda(__isl_take isl_printer *p,
	struct gpu_prog *prog, __isl_keep isl_ast_node *tree,
	struct gpu_types *types, void *user)
{
	struct cuda_info *cuda = user;
	isl_printer *kernel;

	kernel = isl_printer_to_file(isl_printer_get_ctx(p), cuda->kernel_c);
	kernel = isl_printer_set_output_format(kernel, ISL_FORMAT_C);
	kernel = gpu_print_types(kernel, types, prog);
	isl_printer_free(kernel);

	if (!kernel)
		return isl_printer_free(p);

	p = ppcg_start_block(p);

	p = print_cuda_macros(p);

	p = gpu_print_local_declarations(p, prog);
	p = declare_device_arrays(p, prog);
	p = allocate_device_arrays(p, prog);
	
	p = print_host_code(p, prog, tree, cuda);
       
	p = free_device_arrays(p, prog);

	p = ppcg_end_block(p);

	return p;
}

int coding_ants_opt(isl_ctx *ctx, struct ppcg_options *options,
		  const char *input, const char *output)
{
  struct cuda_info cuda;
  struct coding_ants *ca;
  int i;
  int r;

  // must be set so that used_sizes is populated
  options->debug->dump_sizes = 1;
  ca = coding_ants_alloc(input);

  // set the initialization flag
  ca->init = 1;

  // compile and run the original code
  compile(ca, "orig");
  ca->orig_run_time = get_run_time(ca, "orig");
  ca->ctx = ctx;

  cuda.wrap = ca->wrap;

  // perform initial run
  cuda_open_files(&cuda, input, output);  
  r = generate_gpu(ctx, input, cuda.host_c, options, &print_cuda, &cuda, ca);  
  cuda_close_files(&cuda);		

  // get runtime of ppcg default
  compile(ca, "cuda");
  //ca->ppcg_only_time = get_run_time(ca, "ppcg_cuda");
  double ppcg_only = 0;
  for (i = 0; i < 10; ++i) {
    ppcg_only += get_run_time(ca, "ppcg_cuda");
  }
  ca->ppcg_only_time = (ppcg_only / 10.0);
  printf("ppcg only = %0.4e\n", ca->ppcg_only_time);
  
  // initialization complete
  ca->init = 0;
  options->debug->dump_sizes = 0;
  init_graph(ca);

  //ca_apply_optimizations(ca, ant, options);
  if (options->sizes) {
    free(options->sizes);
  }

  options->sizes = (char *)calloc(1000, sizeof(char));

  struct ca_node *reg = NULL;
  reg = optimize(ctx, options, input, output, &print_cuda, ca);
  //r = optimize_random(ctx, options, input, output, &print_cuda, ca, reg);
  //r = optimize_dfs(ctx, options, input, output, &print_cuda, ca);

  //assert (reg != NULL);
  //r = optimize_bfs(ctx, options, input, output, &print_cuda, ca, reg);

  free(options->sizes);
  free(ca);

  return r; 
}


/* Transform the code in the file called "input" by replacing
 * all scops by corresponding CUDA code.
 * The names of the output files are derived from "input".
 *
 * We let generate_gpu do all the hard work and then let it call
 * us back for printing the AST in print_cuda.
 *
 * To prepare for this printing, we first open the output files
 * and we close them after generate_gpu has finished.
 */
int generate_cuda(isl_ctx *ctx, struct ppcg_options *options,
		  const char *input, const char *output)
{
	struct cuda_info cuda;
	int i;
	int r;

	cuda_open_files(&cuda, input, output);

	r = generate_gpu(ctx, input, cuda.host_c, options, &print_cuda, &cuda, NULL);

	cuda_close_files(&cuda);		

	return r;
}
