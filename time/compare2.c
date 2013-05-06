#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<cblas.h>
#include"myblas.h"

int main()
{
    size_t m=10000;
    size_t n=4000;
    //size_t m=100;
    //size_t n=40;
    double *a, *x, *y;
    double alpha=2.0, beta=3.0;
    clock_t t1,t2,t3;
    size_t i;

    a=malloc(sizeof(double)*n*m);
    if(a==NULL){perror("malloc a"); exit(1);}
    for(i=0; i<n*m; i++) a[i]=i;
    x=malloc(sizeof(double)*n);
    if(x==NULL){perror("malloc x"); exit(1);}
    for(i=0; i<n; i++) x[i]=i;
    y=malloc(sizeof(double)*m);
    if(y==NULL){perror("malloc y"); exit(1);}
    for(i=0; i<m; i++) y[i]=i;

    //dgemv(order,trans_a,m,n,alpha,a,lda,x,incx,beta,y,incy)
    // a[0..m][0..n] y[0..m] x[0..n]
    // lda=n
    t1=clock();
    for(i=0; i<10; i++)cblas_dgemv(CblasRowMajor,CblasNoTrans,
	    m,n,alpha,a,n,x,1,beta,y,1);
    t2=clock();
    for(i=0; i<10; i++)my_dgemv(CblasRowMajor,CblasNoTrans,
	    m,n,alpha,a,n,x,1,beta,y,1);
    t3=clock();
    printf("blas dgemv: %g\n", 1.0*n*m*CLOCKS_PER_SEC/(t2-t1));
    printf("my   dgemv: %g\n", 1.0*n*m*CLOCKS_PER_SEC/(t3-t2));
    printf("ratio     : %g\n", (t2-t1)/(double)(t3-t2));

    // transpose
    t1=clock();
    for(i=0; i<10; i++)cblas_dgemv(CblasRowMajor,CblasTrans,
	    m,n,alpha,a,n,y,1,beta,x,1);
    t2=clock();
    for(i=0; i<10; i++)my_dgemv(CblasRowMajor,CblasTrans,
	    m,n,alpha,a,n,y,1,beta,x,1);
    t3=clock();
    printf("blas dgemv: %g\n", 1.0*n*m*CLOCKS_PER_SEC/(t2-t1));
    printf("my   dgemv: %g\n", 1.0*n*m*CLOCKS_PER_SEC/(t3-t2));
    printf("ratio     : %g\n", (t2-t1)/(double)(t3-t2));

    //dger(order, m, n, alpha, dx, incx, dy, incy a, lda)
    // a[0..m][0..n] y[0..m] x[0..n]
    // lda=n
    t1=clock();
    for(i=0; i<10; i++)cblas_dger(CblasRowMajor,m,n,alpha,y,1,x,1,a,n);
    t2=clock();
    for(i=0; i<10; i++)my_dger(CblasRowMajor,m,n,alpha,y,1,x,1,a,n);
    t3=clock();
    printf("blas dger: %g\n", 1.0*n*m*CLOCKS_PER_SEC/(t2-t1));
    printf("my   dger: %g\n", 1.0*n*m*CLOCKS_PER_SEC/(t3-t2));
    printf("ratio    : %g\n", (t2-t1)/(double)(t3-t2));

    free(a);
    free(x);
    free(y);
    return 0;
}
