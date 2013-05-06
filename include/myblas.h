#ifndef MYBLAS
#define MYBLAS

#include<cblas.h>
void my_daxpy(size_t n, double alpha, const double* x, size_t incx,
	double* y, size_t incy);
double my_ddot(size_t n, const double* dx, size_t incx,
	const double* dy, size_t incy);
void my_dscal(size_t n, double alpha, double* x, size_t incx);
size_t my_idamax(size_t n, const double *x, size_t incx);

void my_dgemv(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
	size_t m, size_t n, double alpha, const double* a, size_t lda,
	const double *x, size_t incx, double beta, double* y, size_t incy);

void my_dger(enum CBLAS_ORDER order, size_t m, size_t n, double alpha,
	const double* dx, size_t incx, const double* dy, size_t incy,
	double *a, size_t lda);

void my_dgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
	enum CBLAS_TRANSPOSE trans_b, size_t m, size_t n, size_t k,
	double alpha, const double *a, size_t lda, const double *b, size_t ldb,
	double beta, double *c, size_t ldc);

#endif
