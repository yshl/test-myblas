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

void test_dgemv2()
{
    const size_t m=30, n=45;
    double a[m*n];
    double x[n];
    double y[m];
    double z[m];
    size_t i;
    for(i=0; i<m*n; i++) a[i]=i;
    for(i=0; i<n; i++) x[i]=i+m*n;
    for(i=0; i<m; i++) y[i]=z[i]=i*i;

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

void test_dgemv_trans2()
{
    const size_t m=35, n=45;
    double a[m*n];
    double x[m];
    double y[n];
    double z[n];
    size_t i;
    for(i=0; i<m*n; i++) a[i]=i;
    for(i=0; i<n; i++) x[i]=i+m*n;
    for(i=0; i<m; i++) y[i]=z[i]=i*i;

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

void test_dger2()
{
    const size_t m=35, n=45;
    double a[m*n];
    double b[m*n];
    double x[m];
    double y[n];
    size_t i;
    for(i=0; i<m*n; i++) b[i]=a[i]=i;
    for(i=0; i<m; i++) x[i]=i+m*n;
    for(i=0; i<n; i++) y[i]=i*i;
    my_dger(CblasRowMajor,m,n,2.0,x,1,y,1,a,n);
    cblas_dger(CblasRowMajor,m,n,2.0,x,1,y,1,b,n);
    for(i=0; i<m*n; i++){
	assert(a[i]==b[i]);
    }
}

void test_dgemv_col()
{
    const size_t m=3, n=4;
    double a[3*4]={
	1,2,3,
	4,5,6,
	7,8,9,
	10,11,12
    };
    double x[4]={2,1,4,3};
    double y[3]={6,5,7};
    double z[3]={6,5,7};
    size_t i;
    my_dgemv(CblasColMajor,CblasNoTrans,m,n,2.0,a,m,x,1,2.0,y,1);
    cblas_dgemv(CblasColMajor,CblasNoTrans,m,n,2.0,a,m,x,1,2.0,z,1);
    for(i=0; i<m; i++){
	assert(y[i]==z[i]);
    }
}

void test_dgemv_col_trans()
{
    const size_t m=3, n=4;
    double a[3*4]={
	1,2,3,
	4,5,6,
	7,8,9,
	10,11,12
    };
    double x[3]={2,1,4};
    double y[4]={6,5,7,3};
    double z[4]={6,5,7,3};
    size_t i;
    my_dgemv(CblasColMajor,CblasTrans,m,n,2.0,a,m,x,1,2.0,y,1);
    cblas_dgemv(CblasColMajor,CblasTrans,m,n,2.0,a,m,x,1,2.0,z,1);
    for(i=0; i<m; i++){
	assert(y[i]==z[i]);
    }
}

void test_dger_col()
{
    const size_t m=3, n=4;
    double a[3*4]={
	1,2,3,
	4,5,6,
	7,8,9,
	10,11,12
    };
    double b[3*4];
    double x[3]={2,3,4};
    double y[4]={7,6,4,5};
    size_t i;
    for(i=0; i<m*n; i++) b[i]=a[i];
    my_dger(CblasColMajor,m,n,2.0,x,1,y,1,a,m);
    cblas_dger(CblasColMajor,m,n,2.0,x,1,y,1,b,m);
    for(i=0; i<m*n; i++){
	assert(a[i]==b[i]);
    }
}

int main()
{
    test_dgemv();
    test_dgemv2();
    test_dgemv_trans();
    test_dgemv_trans2();
    test_dger();
    test_dger2();
    test_dgemv_col();
    test_dgemv_col_trans();
    test_dger_col();

    return 0;
}
