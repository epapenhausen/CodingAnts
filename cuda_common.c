/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 */

#include <ctype.h>
#include <limits.h>
#include <string.h>

#include "cuda_common.h"
#include "ppcg.h"

/* Open the host .cu file and the kernel .hu and .cu files for writing.
 * Add the necessary includes.
 */
void cuda_open_files(struct cuda_info *info, const char *input, const char *output)
{
    char name[PATH_MAX];
    char fileName[PATH_MAX];
    char dir[PATH_MAX];
    int len;
    memset(name, 0, PATH_MAX);
    memset(fileName, 0, PATH_MAX);
    memset(dir, 0, PATH_MAX);
    
    // extract the directory of the output file
    char *dirEnd = strrchr(output, '/');
    int dirLen = (dirEnd - output) + 1;
    memcpy(dir, output, dirLen);

    printf("%s\n", input);
    len = ppcg_extract_base_name(fileName, input);
    printf("%s\n", fileName);
    //exit(0);
    strcpy(name, dir);
    strcat(name, fileName);    
    
    len += dirLen;
    
    strcpy(name + len, "_host.cu");
    printf("host name = %s\n", name);
    info->host_c = fopen(name, "w");

    strcpy(name + len, "_kernel.cu");
    info->kernel_c = fopen(name, "w");
    printf("kernel name = %s\n", name);

    strcpy(name + len, "_kernel.hu");
    printf("kernel header = %s\n", name);

    info->kernel_h = fopen(name, "w");
    fprintf(info->host_c, "#include <assert.h>\n");
    fprintf(info->host_c, "#include <stdio.h>\n");
    fprintf(info->host_c, "#include \"%s\"\n", name);

    // add signal handler
    fprintf(info->host_c, "#include <signal.h>\n");    
    fprintf(info->host_c, "void siginthandler(int param)\n");
    fprintf(info->host_c, "{\n");
    fprintf(info->host_c, "\tint i;\n");
    fprintf(info->host_c, "\tint devCount;\n");
    fprintf(info->host_c, "\tcudaGetDeviceCount (&devCount);\n");
    //fprintf(info->host_c, "\tfor (i = 1; i < devCount; ++i)\n");
    //fprintf(info->host_c, "\t{\n");
    fprintf(info->host_c, "\t\tcudaSetDevice(1);\n");
    fprintf(info->host_c, "\t\tcudaDeviceReset();\n");
    //fprintf(info->host_c, "\t}\n");
    fprintf(info->host_c, "\tprintf(\"timeout\\n\");\n");
    fprintf(info->host_c, "\texit(1);\n");
    fprintf(info->host_c, "}\n");

    fprintf(info->kernel_c, "#include \"%s\"\n", name);
    fprintf(info->kernel_h, "#include \"cuda.h\"\n\n");
}

/* Close all output files.
 */
void cuda_close_files(struct cuda_info *info)
{
    fclose(info->kernel_c);
    fclose(info->kernel_h);
    fclose(info->host_c);
}
