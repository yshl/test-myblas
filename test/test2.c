#include<assert.h>
#include<stdlib.h>
#include<cblas.h>
#include"myblas.h"

void test_dgemv()
{
    const size_t m=3, n=4;
    double a[3*4]={
	1,2,3,4,
	5,6,7,8,
	9,10,11,12
    };
    double x[4]={2,1,4,3};
    double y[3]={6,5,7};
    double z[3]={6,5,7};
    size_t i;
    my_dgemv(CblasRowMajor,CblasNoTrans,m,n,2.0,a,n,x,1,2.0,y,1);
    cblas_dgemv(CblasRowMajor,CblasNoTrans,m,n,2.0,a,n,x,1,2.0,z,1);
    for(i=0; i<m; i++){
	assert(y[i]==z[i]);
    }
}

void test_dgemv_trans()
{
    const size_t m=3, n=4;
    double a[3*4]={
	1,2,3,4,
	5,6,7,8,
	9,10,11,12
    };
    double x[3]={2,1,4};
    double y[4]={6,5,7,3};
    double z[4]={6,5,7,3};
    size_t i;
    my_dgemv(CblasRowMajor,CblasTrans,m,n,2.0,a,n,x,1,2.0,y,1);
    cblas_dgemv(CblasRowMajor,CblasTrans,m,n,2.0,a,n,x,1,2.0,z,1);
    for(i=0; i<m; i++){
	assert(y[i]==z[i]);
    }
}

void test_dger()
{
    const size_t m=3, n=4;
    double a[3*4]={
	1,2,3,4,
	5,6,7,8,
	9,10,11,12
    };
    double b[3*4];
    double x[3]={2,3,4};
    double y[4]={7,6,4,5};
    size_t i;
    for(i=0; i<m*n; i++) b[i]=a[i];
    my_dger(CblasRowMajor,m,n,2.0,x,1,y,1,a,n);
    cblas_dger(CblasRowMajor,m,n,2.0,x,1,y,1,b,n);
    for(i=0; i<m*n; i++){
	assert(a[i]==b[i]);
    }
}

int main()
{
    test_dgemv();
    test_dgemv_trans();
    test_dger();

    return 0;
}
