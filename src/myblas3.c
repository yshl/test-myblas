#include<stdio.h>
#include<stdlib.h>
#include<cblas.h>
#include"myblas.h"

static size_t imin(size_t a, size_t b){
    return a<b?a:b;
}

static void dgemm_nn(size_t m, size_t n, size_t k,
	double alpha, const double *a, size_t lda, const double *b, size_t ldb,
	double *c, size_t ldc)
{
    size_t i,j,l;
    const size_t unlp=8;

    for(i=0; i<m; i++){
	for(l=0; l+unlp<=k; l+=unlp){
	    double ail0,ail1,ail2,ail3,ail4,ail5,ail6,ail7;
	    ail0=alpha*a[i*lda+l+0];
	    ail1=alpha*a[i*lda+l+1];
	    ail2=alpha*a[i*lda+l+2];
	    ail3=alpha*a[i*lda+l+3];
	    ail4=alpha*a[i*lda+l+4];
	    ail5=alpha*a[i*lda+l+5];
	    ail6=alpha*a[i*lda+l+6];
	    ail7=alpha*a[i*lda+l+7];
	    for(j=0; j<n; j++){
		c[i*ldc+j]+=ail0*b[(l+0)*ldb+j]
		    +ail1*b[(l+1)*ldb+j]
		    +ail2*b[(l+2)*ldb+j]
		    +ail3*b[(l+3)*ldb+j]
		    +ail4*b[(l+4)*ldb+j]
		    +ail5*b[(l+5)*ldb+j]
		    +ail6*b[(l+6)*ldb+j]
		    +ail7*b[(l+7)*ldb+j];
	    }
	}
	for(; l<k; l++){
	    double ail=alpha*a[i*lda+l];
	    for(j=0; j<n; j++){
		c[i*ldc+j]+=ail*b[l*ldb+j];
	    }
	}
    }
}

static void my_dgemm_NN(size_t m, size_t n, size_t k,
	double alpha, const double *a, size_t lda, const double *b, size_t ldb,
	double *c, size_t ldc)
{

    // c[0:m][0:n], a[0:m][0:k] b[0:k][0:n]
    // c[(0:m)*ldc+(0:n)], a[(0:m)*lda+(0:k)] b[(0:k)*ldb+(0:n)]
    size_t i,j,l;
    const size_t blocksize=112;

    for(i=0; i<m; i+=blocksize){
	size_t iend=imin(blocksize, m-i);
	for(l=0; l<k; l+=blocksize){
	    size_t lend=imin(blocksize, k-l);
	    for(j=0; j<n; j+=blocksize){
		size_t jend=imin(blocksize, n-j);
		dgemm_nn(iend,jend,lend,alpha,a+i*lda+l,lda,b+l*ldb+j,ldb,
			c+i*ldc+j,ldc);
	    }
	}
    }
}

static void dgemm_nt(size_t m, size_t n, size_t k,
	double alpha, const double *a, size_t lda, const double *b, size_t ldb,
	double *c, size_t ldc)
{
    size_t i,j,l;
    const size_t unlp=6;
    for(i=0; i+unlp<=m; i+=unlp){
	for(j=0; j<n; j++){
	    double sum0=0.0,sum1=0.0,sum2=0.0,sum3=0.0,
		   sum4=0.0,sum5=0.0;
	    for(l=0; l<k; l++){
		double bjl=b[j*ldb+l];
		sum0+=a[(i+0)*lda+l]*bjl;
		sum1+=a[(i+1)*lda+l]*bjl;
		sum2+=a[(i+2)*lda+l]*bjl;
		sum3+=a[(i+3)*lda+l]*bjl;
		sum4+=a[(i+4)*lda+l]*bjl;
		sum5+=a[(i+5)*lda+l]*bjl;
	    }
	    c[(i+0)*ldc+j]+=alpha*sum0;
	    c[(i+1)*ldc+j]+=alpha*sum1;
	    c[(i+2)*ldc+j]+=alpha*sum2;
	    c[(i+3)*ldc+j]+=alpha*sum3;
	    c[(i+4)*ldc+j]+=alpha*sum4;
	    c[(i+5)*ldc+j]+=alpha*sum5;
	}
    }
    for(; i<m; i++){
	for(j=0; j<n; j++){
	    double sum=0.0;
	    for(l=0; l<k; l++){
		sum+=a[i*lda+l]*b[j*ldb+l];
	    }
	    c[i*ldc+j]+=alpha*sum;
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
    const size_t blocksize=96;
    for(i=0; i<m; i+=blocksize){
	size_t iend=imin(blocksize,m-i);
	for(l=0; l<k; l+=blocksize){
	    size_t lend=imin(blocksize,k-l);
	    for(j=0; j<n; j+=blocksize){
		size_t jend=imin(blocksize,n-j);
		dgemm_nt(iend,jend,lend,alpha,a+i*lda+l,lda,b+j*ldb+l,ldb,
			c+i*ldc+j,ldc);
	    }
	}
    }
}

static void dgemm_tn(size_t m, size_t n, size_t k,
	double alpha, const double *a, size_t lda, const double *b, size_t ldb,
	double *c, size_t ldc)
{
    size_t i,j,l;
    const size_t unlp=8;
    for(l=0; l+unlp<=k; l+=unlp){
	for(i=0; i<m; i++){
	    double ali0=alpha*a[(l+0)*lda+i];
	    double ali1=alpha*a[(l+1)*lda+i];
	    double ali2=alpha*a[(l+2)*lda+i];
	    double ali3=alpha*a[(l+3)*lda+i];
	    double ali4=alpha*a[(l+4)*lda+i];
	    double ali5=alpha*a[(l+5)*lda+i];
	    double ali6=alpha*a[(l+6)*lda+i];
	    double ali7=alpha*a[(l+7)*lda+i];
	    for(j=0; j<n; j++){
		c[i*ldc+j]+=ali0*b[(l+0)*ldb+j]
		    +ali1*b[(l+1)*ldb+j]
		    +ali2*b[(l+2)*ldb+j]
		    +ali3*b[(l+3)*ldb+j]
		    +ali4*b[(l+4)*ldb+j]
		    +ali5*b[(l+5)*ldb+j]
		    +ali6*b[(l+6)*ldb+j]
		    +ali7*b[(l+7)*ldb+j];
	    }
	}
    }
    for(; l<k; l++){
	for(i=0; i<m; i++){
	    double ali=alpha*a[l*lda+i];
	    for(j=0; j<n; j++){
		c[i*ldc+j]+=ali*b[l*ldb+j];
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
    const size_t blocksize=96;
    for(l=0; l<k; l+=blocksize){
	size_t lend=imin(blocksize,k-l);
	for(i=0; i<m; i+=blocksize){
	    size_t iend=imin(blocksize,m-i);
	    for(j=0; j<n; j+=blocksize){
		size_t jend=imin(blocksize,n-j);
		dgemm_tn(iend,jend,lend,alpha,a+l*lda+i,lda,b+l*ldb+j,ldb,
			c+i*ldc+j,ldc);
	    }
	}
    }
}

static void dgemm_tt(size_t m, size_t n, size_t k,
	double alpha, const double *a, size_t lda, const double *b, size_t ldb,
	double *c, size_t ldc)
{
    size_t i,j,l;
    const size_t unlp=4;
    for(i=0; i<m; i++){
	for(j=0; j+unlp<=n; j+=unlp){
	    double sum0=0.0;
	    double sum1=0.0;
	    double sum2=0.0;
	    double sum3=0.0;
	    for(l=0; l<k; l++){
		double ali=a[l*lda+i];
		sum0+=ali*b[(j+0)*ldb+l];
		sum1+=ali*b[(j+1)*ldb+l];
		sum2+=ali*b[(j+2)*ldb+l];
		sum3+=ali*b[(j+3)*ldb+l];
	    }
	    c[i*ldc+j+0]+=alpha*sum0;
	    c[i*ldc+j+1]+=alpha*sum1;
	    c[i*ldc+j+2]+=alpha*sum2;
	    c[i*ldc+j+3]+=alpha*sum3;
	}
	for(; j<n; j++){
	    double sum=0.0;
	    for(l=0; l<k; l++){
		sum+=a[l*lda+i]*b[j*ldb+l];
	    }
	    c[i*ldc+j]+=alpha*sum;
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
    const size_t blocksize=144;
    for(i=0; i<m; i+=blocksize){
	size_t iend=imin(blocksize,m-i);
	for(l=0; l<k; l+=blocksize){
	    size_t lend=imin(blocksize,k-l);
	    for(j=0; j<n; j+=blocksize){
		size_t jend=imin(blocksize,n-j);
		dgemm_tt(iend,jend,lend,alpha,a+l*lda+i,lda,b+j*ldb+l,ldb,
			c+i*ldc+j,ldc);
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
