#include<stdio.h>
#include<stdlib.h>
#include<cblas.h>
#include"myblas.h"

static void my_dgemv_RN(size_t m, size_t n, double alpha,
	const double* a, size_t lda, const double *x, size_t incx,
	double beta, double* y, size_t incy)
{
    // y[i]=beta*y[i]+alpha*a[i,j]*x[j]
    // y[i]=beta*y[i]+alpha*a[i*lda+j]*x[j]
    size_t i,j;
    const size_t unlp=4;
    for(i=0; i+unlp<=m; i+=unlp){
	double sum[unlp];
	for(j=0; j<unlp; j++) sum[j]=0.0;
	for(j=0; j<n; j++){
	    double xj=x[j*incx];
	    sum[0]+=a[(i+0)*lda+j]*xj;
	    sum[1]+=a[(i+1)*lda+j]*xj;
	    sum[2]+=a[(i+2)*lda+j]*xj;
	    sum[3]+=a[(i+3)*lda+j]*xj;
	}
	for(j=0; j<unlp; j++){
	    y[(i+j)*incy]=beta*y[(i+j)*incy]+alpha*sum[j];
	}
    }
    for(; i<m; i++){
	y[i*incy]=beta*y[i*incy]+alpha*my_ddot(n,&a[i*lda],1,x,incx);
    }
}

static void my_dgemv_RT(size_t m, size_t n, double alpha,
	const double* a, size_t lda, const double *x, size_t incx,
	double beta, double* y, size_t incy)
{
    // y[j]=beta*y[j]+alpha*a[i,j]*x[i]
    // y[j]=beta*y[j]+alpha*a[i*lda+j]*x[i]
    size_t i,j;
    const size_t unlp=6;
    my_dscal(n,beta,y,incy);
    for(i=0; i+unlp<=m; i+=unlp){
	double ax[unlp];
	for(j=0; j<unlp; j++) ax[j]=alpha*x[(i+j)*incx];
	for(j=0; j<n; j++){
	    y[j*incy]+=ax[0]*a[(i+0)*lda+j]
		+ax[1]*a[(i+1)*lda+j]
		+ax[2]*a[(i+2)*lda+j]
		+ax[3]*a[(i+3)*lda+j]
		+ax[4]*a[(i+4)*lda+j]
		+ax[5]*a[(i+5)*lda+j];
	}
    }
    for(; i<m; i++){
	my_daxpy(n,alpha*x[i*incx],&a[i*lda],1,y,incy);
    }
}

void my_dgemv(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
	size_t m, size_t n, double alpha, const double* a, size_t lda,
	const double *x, size_t incx, double beta, double* y, size_t incy)
{
    // y=beta*y+alpha*a*x
    if(alpha==0.0){
	if(trans_a==CblasNoTrans){
	    my_dscal(m, beta, y, incy);
	}else{
	    my_dscal(n, beta, y, incy);
	}
	return;
    }
    if(order==CblasRowMajor){
	// a[i,j]=a[i*lda+j]
	// 0<=i<m, 0<=j<n
	if(trans_a==CblasNoTrans){
	    // y[i]=beta*y[i]+alpha*a[i,j]*x[j]
	    // y[i]=beta*y[i]+alpha*a[i*lda+j]*x[j]
	    my_dgemv_RN(m,n,alpha,a,lda,x,incx,beta,y,incy);
	}else{
	    // y[j]=beta*y[j]+alpha*a[i,j]*x[i]
	    // y[j]=beta*y[j]+alpha*a[i*lda+j]*x[i]
	    my_dgemv_RT(m,n,alpha,a,lda,x,incx,beta,y,incy);
	}
    }else{
	// a[i,j]=a[j*lda+i]
	// 0<=i<m, 0<=j<n
	if(trans_a==CblasNoTrans){
	    // y[i]=beta*y[i]+alpha*a[i,j]*x[j]
	    // y[i]=beta*y[i]+alpha*a[j*lda+i]*x[j]
	    my_dgemv_RT(n,m,alpha,a,lda,x,incx,beta,y,incy);
	}else{
	    // y[j]=beta*y[j]+alpha*a[i,j]*x[i]
	    // y[j]=beta*y[j]+alpha*a[j*lda+i]*x[i]
	    my_dgemv_RN(n,m,alpha,a,lda,x,incx,beta,y,incy);
	}
    }
}

static void my_dger_R(size_t m, size_t n, double alpha,
	const double* dx, size_t incx, const double* dy, size_t incy,
	double *a, size_t lda)
{
    // a[i,j]=a[i*lda+j]
    // a[i*lda+j]+=alpha*x[i]*y[j]
    size_t i,j;
    const size_t unlp=4;
    for(i=0; i+unlp<=m; i+=unlp){
	double axi[unlp];
	for(j=0; j<unlp; j++) axi[j]=alpha*dx[(i+j)*incx];
	for(j=0; j<n; j++){
	    double yj=dy[j*incy];
	    a[(i+0)*lda+j]+=axi[0]*yj;
	    a[(i+1)*lda+j]+=axi[1]*yj;
	    a[(i+2)*lda+j]+=axi[2]*yj;
	    a[(i+3)*lda+j]+=axi[3]*yj;
	}
    }
    for(; i<m; i++){
	my_daxpy(n,alpha*dx[i*incx],dy,incy,&a[i*lda],1);
    }
}

void my_dger(enum CBLAS_ORDER order, size_t m, size_t n, double alpha,
	const double* dx, size_t incx, const double* dy, size_t incy,
	double *a, size_t lda)
{
    // a[lda][]
    // a+=alpha*x*trans(y)
    if(alpha==0.0) return;
    if(order==CblasRowMajor){
	// a[i,j]=a[i*lda+j]
	// a[i*lda+j]+=alpha*x[i]*y[j]
	my_dger_R(m,n,alpha,dx,incx,dy,incy,a,lda);
    }else if(order==CblasColMajor){
	// a[i,j]=a[j*lda+i]
	// a[j*lda+i]+=alpha*x[i]*y[j]
	my_dger_R(n,m,alpha,dy,incy,dx,incx,a,lda);
    }else{
	fprintf(stderr, "Illegal argument order %d\n", order);
	exit(1);
    }
}
