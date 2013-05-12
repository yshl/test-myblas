#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<cblas.h>
#include"myblas.h"

int main(int argc, char *argv[])
{
    size_t m=10000;
    size_t n=4000;
    double *a, *x, *y;
    double alpha=2.0, beta=3.0;
    clock_t t1,t2,t3;
    size_t i;
    FILE *logfile;

    logfile=stdout;
    if(argc>1){
	logfile=fopen(argv[1],"w");
	if(logfile==NULL){
	    perror(argv[1]);
	    exit(1);
	}
    }

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
    fprintf(logfile,"blas dgemv: %g\n", 10.0*n*m*CLOCKS_PER_SEC/(t2-t1));
    fprintf(logfile,"my   dgemv: %g\n", 10.0*n*m*CLOCKS_PER_SEC/(t3-t2));
    fprintf(logfile,"ratio     : %g\n", (t2-t1)/(double)(t3-t2));

    // transpose
    t1=clock();
    for(i=0; i<10; i++)cblas_dgemv(CblasRowMajor,CblasTrans,
	    m,n,alpha,a,n,y,1,beta,x,1);
    t2=clock();
    for(i=0; i<10; i++)my_dgemv(CblasRowMajor,CblasTrans,
	    m,n,alpha,a,n,y,1,beta,x,1);
    t3=clock();
    fprintf(logfile,"blas dgemv T: %g\n", 10.0*n*m*CLOCKS_PER_SEC/(t2-t1));
    fprintf(logfile,"my   dgemv T: %g\n", 10.0*n*m*CLOCKS_PER_SEC/(t3-t2));
    fprintf(logfile,"ratio     : %g\n", (t2-t1)/(double)(t3-t2));

    //dger(order, m, n, alpha, dx, incx, dy, incy a, lda)
    // a[0..m][0..n] y[0..m] x[0..n]
    // lda=n
    t1=clock();
    for(i=0; i<10; i++)cblas_dger(CblasRowMajor,m,n,alpha,y,1,x,1,a,n);
    t2=clock();
    for(i=0; i<10; i++)my_dger(CblasRowMajor,m,n,alpha,y,1,x,1,a,n);
    t3=clock();
    fprintf(logfile,"blas dger: %g\n", 10.0*n*m*CLOCKS_PER_SEC/(t2-t1));
    fprintf(logfile,"my   dger: %g\n", 10.0*n*m*CLOCKS_PER_SEC/(t3-t2));
    fprintf(logfile,"ratio    : %g\n", (t2-t1)/(double)(t3-t2));

    free(a);
    free(x);
    free(y);
    fclose(logfile);
    return 0;
}
