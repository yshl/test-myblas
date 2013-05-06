#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<cblas.h>
#include"myblas.h"

int main()
{
    size_t n=20000000;
    //size_t n=100;
    double *x, *y;
    double alpha=2.0;
    clock_t t1,t2,t3;
    size_t i;

    x=malloc(sizeof(double)*n);
    if(x==NULL){perror("malloc x"); exit(1);}
    y=malloc(sizeof(double)*n);
    if(y==NULL){perror("malloc y"); exit(1);}
    for(i=0; i<n; i++){
	x[i]=i;
	y[i]=i;
    }

    t1=clock();
    for(i=0; i<10; i++)cblas_daxpy(n,alpha,x,1,y,1);
    t2=clock();
    for(i=0; i<10; i++)my_daxpy(n,alpha,x,1,y,1);
    t3=clock();
    printf("blas daxpy: %g\n", 1.0*n*CLOCKS_PER_SEC/(t2-t1));
    printf("my   daxpy: %g\n", 1.0*n*CLOCKS_PER_SEC/(t3-t2));
    printf("ratio     : %g\n", (t2-t1)/(double)(t3-t2));

    t1=clock();
    for(i=0; i<10; i++){x[0]+=cblas_ddot(n,x,1,y,1);}
    t2=clock();
    for(i=0; i<10; i++){x[0]+=my_ddot(n,x,1,y,1);}
    t3=clock();
    printf("blas ddot: %g\n", 1.0*n*CLOCKS_PER_SEC/(t2-t1));
    printf("my   ddot: %g\n", 1.0*n*CLOCKS_PER_SEC/(t3-t2));
    printf("ratio    : %g\n", (t2-t1)/(double)(t3-t2));

    t1=clock();
    for(i=0; i<10; i++)cblas_dscal(n,alpha,x,1);
    t2=clock();
    for(i=0; i<10; i++)my_dscal(n,alpha,x,1);
    t3=clock();
    printf("blas dscal: %g\n", 1.0*n*CLOCKS_PER_SEC/(t2-t1));
    printf("my   dscal: %g\n", 1.0*n*CLOCKS_PER_SEC/(t3-t2));
    printf("ratio     : %g\n", (t2-t1)/(double)(t3-t2));

    t1=clock();
    for(i=0; i<10; i++){x[0]+=cblas_idamax(n,x,1);}
    t2=clock();
    for(i=0; i<10; i++){x[0]+=my_idamax(n,x,1);}
    t3=clock();
    printf("blas idamax: %g\n", 1.0*n*CLOCKS_PER_SEC/(t2-t1));
    printf("my   idamax: %g\n", 1.0*n*CLOCKS_PER_SEC/(t3-t2));
    printf("ratio      : %g\n", (t2-t1)/(double)(t3-t2));

    free(x);
    free(y);
    return 0;
}
