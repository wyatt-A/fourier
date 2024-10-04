#include <stdio.h>
#include <fftw3.h>
#include <omp.h>
#include<string.h>

int execute(int rank, const int *n, int sign, const char *wisdom_string_in, char **wisdom_string_out, fftwf_complex *in, fftwf_complex *out) {

    fftwf_plan plan;

    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());

    if (wisdom_string_in != NULL) {
        int import_status = fftwf_import_wisdom_from_string(wisdom_string_in);
        if (import_status == 0) {
            printf("failed to recover wisdom from string\n");
        }
    }else {
        printf("input wisodm string is null\n");
    }

    plan = fftwf_plan_dft(rank, n, in, out, sign, FFTW_MEASURE | FFTW_WISDOM_ONLY);
    //plan = fftwf_plan_dft(rank, n, in, out, sign, FFTW_ESTIMATE | FFTW_WISDOM_ONLY);

    int status = 0;
    if (!plan) {
        printf("no wisdom found, measuring optimal plan ...\n");
        int product = 1;
        for (int i=0;i<rank;i++) {
            product = product * n[i];
        }
        fftwf_complex *tmp = fftwf_malloc(sizeof(fftwf_complex) * product);
        memcpy(tmp, in, sizeof(fftwf_complex) * product);
        plan = fftwf_plan_dft(rank, n, tmp, tmp, sign, FFTW_MEASURE);
        fftwf_free(tmp);
        status = -1;
    }else {
        fftwf_execute(plan);
    }

    // export wisdom from the run
    *wisdom_string_out = fftwf_export_wisdom_to_string();

    fftwf_destroy_plan(plan);
    fftwf_cleanup_threads();

    // return -1 if we had to measure the optimal fft and no xform was run
    // it means we have to re-run the xfrom with the new wisdom
    return status;
}

char * export_string_to_rust(void) {
    char *c_string = (char *)malloc(100);  // Allocate 100 bytes
    if (c_string != NULL) {
        strcpy(c_string, "Hello from C!");  // Fill the string
        return c_string;
    }else {
        printf("failed to allocate string in c\n");
        return NULL;
    }
}

int export_string_to_rust2(char ** string) {
    char *c_string = (char *)malloc(100);  // Allocate 100 bytes
    if (c_string != NULL) {
        strcpy(c_string, "Hello from C!");  // Fill the string
        *string = c_string;
        return 0;
    }else {
        printf("failed to allocate string in c\n");
        return 1;
    }
}

void free_c_string(char * string) {
    free(string);
}

int import_string_from_rust(const char * string) {

    printf("in c function\n");

    if (string != NULL) {
        printf("string from c: %s\n",string);
        return 0;
    }else {
        printf("string is null\n");
        return 1;
    }
}

int test_get_wisdom(char ** wisdom_string) {

    fftwf_plan plan;

    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());

    fftwf_complex *tmp = fftwf_malloc(sizeof(fftwf_complex) * 512);
    int n = 512;
    plan = fftwf_plan_dft(1, &n, tmp, tmp, -1, FFTW_MEASURE);

    *wisdom_string = fftwf_export_wisdom_to_string();

    printf("wis: %s\n",*wisdom_string);

    fftwf_free(tmp);
    fftwf_destroy_plan(plan);
    fftwf_cleanup_threads();
    return 0;
}