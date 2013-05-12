#include<stdio.h>
#include<stdlib.h>
#include<cblas.h>
#include"myblas.h"

static size_t imin(size_t a, size_t b){
    return a<b?a:b;
}

static void my_dgemm_NN(size_t m, size_t n, size_t k,
	double alpha, const double *a, size_t lda, const double *b, size_t ldb,
	double *c, size_t ldc)
{

    // c[0:m][0:n], a[0:m][0:k] b[0:k][0:n]
    // c[(0:m)*ldc+(0:n)], a[(0:m)*lda+(0:k)] b[(0:k)*ldb+(0:n)]
    size_t i,j,l;
    const size_t blocksize=48;
    double tmp[blocksize][blocksize];

    for(i=0; i<m; i+=blocksize){
	size_t iend=imin(i+blocksize, m);
	for(l=0; l<k; l+=blocksize){
	    size_t lend=imin(l+blocksize, k);
	    size_t ii,jj,ll;
	    for(ii=i; ii<iend; ii++){
		for(ll=l; ll<lend; ll++){
		    tmp[ii-i][ll-l]=alpha*a[ii*lda+ll];
		}
	    }
	    for(j=0; j<n; j+=blocksize){
		size_t jend=imin(j+blocksize, n);
		for(ii=i; ii<iend; ii++){
		    for(ll=l; ll<lend; ll++){
			for(jj=j; jj<jend; jj++){
			    c[ii*ldc+jj]+=tmp[ii-i][ll-l]*b[ll*ldb+jj];
			}
		    }
		}
	    }
	}
    }
}

static void my_dgemm_NT(size_t m, size_t n, size_t k,
	double alpha, const double *a, size_t lda, const double *b, size_t ldb,
	double *c, size_t ldc)
{

    // c[0:m][0:n], a[0:m][0:k] b[0:n][0:k]
    // c[(0:m)*ldc+(0:n)], a[(0:m)*lda+(0:k)] b[(0:n)*ldb+(0:k)]
    size_t i,j,l;
    const size_t blocksize=48;
    double tmp[blocksize][blocksize];
    for(i=0; i<m; i+=blocksize){
	size_t iend=imin(i+blocksize,m);
	for(l=0; l<k; l+=blocksize){
	    size_t lend=imin(l+blocksize,k);
	    size_t ii,jj,ll;
	    for(ii=i; ii<iend; ii++){
		for(ll=l; ll<lend; ll++){
		    tmp[ii-i][ll-l]=a[ii*lda+ll];
		}
	    }
	    for(j=0; j<n; j+=blocksize){
		size_t jend=imin(j+blocksize,n);
		for(ii=i; ii<iend; ii++){
		    for(jj=j; jj<jend; jj++){
			double sum=0.0;
			for(ll=l; ll<lend; ll++){
			    sum+=tmp[ii-i][ll-l]*b[jj*ldb+ll];
			}
			c[ii*ldc+jj]+=alpha*sum;
		    }
		}
	    }
	}
    }
}

static void my_dgemm_TN(size_t m, size_t n, size_t k,
	double alpha, const double *a, size_t lda, const double *b, size_t ldb,
	double *c, size_t ldc)
{
    // c[0:m][0:n], a[0:k][0:m] b[0:k][0:n]
    // c[(0:m)*ldc+(0:n)], a[(0:k)*lda+(0:m)] b[(0:k)*ldb+(0:n)]
    size_t i,j,l;
    const size_t blocksize=48;
    double tmp[blocksize][blocksize];
    for(l=0; l<k; l+=blocksize){
	size_t lend=imin(l+blocksize,k);
	for(i=0; i<m; i+=blocksize){
	    size_t iend=imin(i+blocksize,m);
	    size_t ii,jj,ll;
	    for(ll=l; ll<lend; ll++){
		for(ii=i; ii<iend; ii++){
		    tmp[ll-l][ii-i]=alpha*a[ll*lda+ii];
		}
	    }
	    for(j=0; j<n; j+=blocksize){
		size_t jend=imin(j+blocksize,n);
		for(ll=l; ll<lend; ll++){
		    for(ii=i; ii<iend; ii++){
			for(jj=j; jj<jend; jj++){
			    c[ii*ldc+jj]+=tmp[ll-l][ii-i]*b[ll*ldb+jj];
			}
		    }
		}
	    }
	}
    }
}

static void my_dgemm_TT(size_t m, size_t n, size_t k,
	double alpha, const double *a, size_t lda, const double *b, size_t ldb,
	double *c, size_t ldc)
{
    // c[0:m][0:n], a[0:k][0:m] b[0:n][0:k]
    // c[(0:m)*ldc+(0:n)], a[(0:k)*lda+(0:m)] b[(0:n)*ldb+(0:k)]
    size_t i,j,l;
    const size_t blocksize=48;
    double tmp[blocksize][blocksize];
    for(i=0; i<m; i+=blocksize){
	size_t iend=imin(i+blocksize,m);
	for(l=0; l<k; l+=blocksize){
	    size_t lend=imin(l+blocksize,k);
	    size_t ii,jj,ll;
	    for(ii=i; ii<iend; ii++){
		for(ll=l; ll<lend; ll++){
		    tmp[ii-i][ll-l]=a[ll*lda+ii];
		}
	    }
	    for(j=0; j<n; j+=blocksize){
		size_t jend=imin(j+blocksize,n);
		for(ii=i; ii<iend; ii++){
		    for(jj=j; jj<jend; jj++){
			double sum=0.0;
			for(ll=l; ll<lend; ll++){
			    sum+=tmp[ii-i][ll-l]*b[jj*ldb+ll];
			}
			c[ii*ldc+jj]+=alpha*sum;
		    }
		}
	    }
	}
    }
}

void my_dgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
	enum CBLAS_TRANSPOSE trans_b, size_t m, size_t n, size_t k,
	double alpha, const double *a, size_t lda, const double *b, size_t ldb,
	double beta, double *c, size_t ldc)
{
    // c=beta*c+alpha*a*b
    // c[0:m][0:n]
    if(order==CblasRowMajor){
	if(beta!=1.0){
	    size_t i;
	    for(i=0; i<m; i++){
		my_dscal(n,beta,&c[i*ldc],1);
	    }
	}
	if(alpha==0.0){
	    return;
	}
	if(trans_a==CblasNoTrans){
	    if(trans_b==CblasNoTrans){
		my_dgemm_NN(m,n,k,alpha,a,lda,b,ldb,c,ldc);
	    }else{
		my_dgemm_NT(m,n,k,alpha,a,lda,b,ldb,c,ldc);
	    }
	}else{
	    if(trans_b==CblasNoTrans){
		my_dgemm_TN(m,n,k,alpha,a,lda,b,ldb,c,ldc);
	    }else{
		my_dgemm_TT(m,n,k,alpha,a,lda,b,ldb,c,ldc);
	    }
	}
    }else{
	my_dgemm(CblasRowMajor,trans_b,trans_a,n,m,k,alpha,b,ldb,a,lda,beta,
		c,ldc);
    }
}
