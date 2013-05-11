#include<assert.h>
#include<stdlib.h>
#include<cblas.h>
#include"myblas.h"

void test_dgemm_NN()
{
    const size_t m=2,n=3,k=4;
    double a[2*4]={
	1,2,3,4,
        5,6,7,8,
    };
    double b[4*3]={
	9,10,11,
	12,13,14,
	15,16,17,
	18,19,20,
    };
    double c[2*3]={
	3,2,1,
	6,5,4,
    };
    double d[2*3];
    size_t i;
    for(i=0; i<m*n; i++) d[i]=c[i];

    my_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
	    m,n,k,2.0,a,k,b,n,2.0,c,n);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
	    m,n,k,2.0,a,k,b,n,2.0,d,n);
    for(i=0; i<m*n; i++){
	assert(c[i]==d[i]);
    }
}

void test_dgemm_NT()
{
    const size_t m=2,n=3,k=4;
    double a[2*4]={
	1,2,3,4,
        5,6,7,8,
    };
    double b[3*4]={
	9,10,11,12,
	13,14,15,16,
	17,18,19,20,
    };
    double c[2*3]={
	3,2,1,
	6,5,4,
    };
    double d[2*3];
    size_t i;
    for(i=0; i<m*n; i++) d[i]=c[i];

    my_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
	    m,n,k,2.0,a,k,b,k,2.0,c,n);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
	    m,n,k,2.0,a,k,b,k,2.0,d,n);
    for(i=0; i<m*n; i++){
	assert(c[i]==d[i]);
    }
}

void test_dgemm_TN()
{
    const size_t m=2,n=3,k=4;
    double a[4*2]={
	1,2,
	3,4,
        5,6,
	7,8,
    };
    double b[4*3]={
	9,10,11,
	12,13,14,
	15,16,17,
	18,19,20,
    };
    double c[2*3]={
	3,2,1,
	6,5,4,
    };
    double d[2*3];
    size_t i;
    for(i=0; i<m*n; i++) d[i]=c[i];

    my_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
	    m,n,k,2.0,a,m,b,n,2.0,c,n);
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
	    m,n,k,2.0,a,m,b,n,2.0,d,n);
    for(i=0; i<m*n; i++){
	assert(c[i]==d[i]);
    }
}

void test_dgemm_TT()
{
    const size_t m=2,n=3,k=4;
    double a[4*2]={
	1,2,
	3,4,
        5,6,
	7,8,
    };
    double b[3*4]={
	9,10,11,12,
	13,14,15,16,
	17,18,19,20,
    };
    double c[2*3]={
	3,2,1,
	6,5,4,
    };
    double d[2*3];
    size_t i;
    for(i=0; i<m*n; i++) d[i]=c[i];

    my_dgemm(CblasRowMajor,CblasTrans,CblasTrans,
	    m,n,k,2.0,a,m,b,k,2.0,c,n);
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,
	    m,n,k,2.0,a,m,b,k,2.0,d,n);
    for(i=0; i<m*n; i++){
	assert(c[i]==d[i]);
    }
}

void test_dgemm_NN_col()
{
    const size_t m=2,n=3,k=4;
    double a[2*4]={
	1,2,
	3,4,
        5,6,
	7,8,
    };
    double b[4*3]={
	9,10,11,12,
	13,14,15,16,
	17,18,19,20,
    };
    double c[2*3]={
	3,2,
	1,6,
	5,4,
    };
    double d[2*3];
    size_t i;
    for(i=0; i<m*n; i++) d[i]=c[i];

    my_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
	    m,n,k,2.0,a,m,b,k,2.0,c,m);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
	    m,n,k,2.0,a,m,b,k,2.0,d,m);
    for(i=0; i<m*n; i++){
	assert(c[i]==d[i]);
    }
}

int main()
{
    test_dgemm_NN();
    test_dgemm_NT();
    test_dgemm_TN();
    test_dgemm_TT();
    test_dgemm_NN_col();

    return 0;
}
