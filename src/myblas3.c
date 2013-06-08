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
	    size_t i1;
	    double ail[unlp];
	    for(i1=0; i1<unlp; i1++){
		ail[i1]=alpha*a[i*lda+l+i1];
	    }
	    for(j=0; j<n; j++){
		c[i*ldc+j]+=ail[0]*b[(l+0)*ldb+j]
		    +ail[1]*b[(l+1)*ldb+j]
		    +ail[2]*b[(l+2)*ldb+j]
		    +ail[3]*b[(l+3)*ldb+j]
		    +ail[4]*b[(l+4)*ldb+j]
		    +ail[5]*b[(l+5)*ldb+j]
		    +ail[6]*b[(l+6)*ldb+j]
		    +ail[7]*b[(l+7)*ldb+j];
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
    const size_t unlp1=12, unlp2=6;
    for(i=0; i+unlp1<=m; i+=unlp1){
	for(j=0; j<n; j++){
	    size_t i1;
	    double sum[unlp1];
	    for(i1=0; i1<unlp1; i1++){
		sum[i1]=0.0;
	    }
	    for(l=0; l+unlp2<=k; l+=unlp2){
		size_t l2;
		for(i1=0; i1<unlp1; i1++){
		    for(l2=0; l2<unlp2; l2++){
			sum[i1]+=a[(i+i1)*lda+l+l2]*b[j*ldb+l+l2];
		    }
		}
	    }
	    for(; l<k; l++){
		double bjl=b[j*ldb+l];
		for(i1=0; i1<unlp1; i1++){
		    sum[i1]+=a[(i+i1)*lda+l]*bjl;
		}
	    }
	    for(i1=0; i1<unlp1; i1++){
		c[(i+i1)*ldc+j]+=alpha*sum[i1];
	    }
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
	    double ali[unlp];
	    size_t i1;
	    for(i1=0; i1<unlp; i1++){
		ali[i1]=alpha*a[(l+i1)*lda+i];
	    }
	    for(j=0; j<n; j++){
		c[i*ldc+j]+=ali[0]*b[(l+0)*ldb+j]
		    +ali[1]*b[(l+1)*ldb+j]
		    +ali[2]*b[(l+2)*ldb+j]
		    +ali[3]*b[(l+3)*ldb+j]
		    +ali[4]*b[(l+4)*ldb+j]
		    +ali[5]*b[(l+5)*ldb+j]
		    +ali[6]*b[(l+6)*ldb+j]
		    +ali[7]*b[(l+7)*ldb+j];
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
    size_t unlp=8;
    for(j=0; j<n; j++){
	for(l=0; l+unlp<=k; l+=unlp){
	    double bjl[unlp];
	    size_t l1;
	    for(l1=0; l1<unlp; l1++){
		bjl[l1]=alpha*b[j*ldb+l+l1];
	    }
	    for(i=0; i<m; i++){
		c[i*ldc+j]+=a[(l+0)*lda+i]*bjl[0]
		    +a[(l+1)*lda+i]*bjl[1]
		    +a[(l+2)*lda+i]*bjl[2]
		    +a[(l+3)*lda+i]*bjl[3]
		    +a[(l+4)*lda+i]*bjl[4]
		    +a[(l+5)*lda+i]*bjl[5]
		    +a[(l+6)*lda+i]*bjl[6]
		    +a[(l+7)*lda+i]*bjl[7];
	    }
	}
	for(; l<k; l++){
	    double bjl=alpha*b[j*ldb+l];
	    for(i=0; i<m; i++){
		c[i*ldc+j]+=a[l*lda+i]*bjl;
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
