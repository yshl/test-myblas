#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<cblas.h>
#include"myblas.h"

int main(int argc, char *argv[])
{
    size_t m=1000;
    size_t n=200;
    size_t k=500;
    double *a, *b, *c;
    double alpha=1.0, beta=0.0;
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

    a=malloc(sizeof(double)*m*k);
    if(a==NULL){perror("malloc a"); exit(1);}
    for(i=0; i<m*k; i++) a[i]=i;
    b=malloc(sizeof(double)*k*n);
    if(b==NULL){perror("malloc b"); exit(1);}
    for(i=0; i<k*n; i++) b[i]=i;
    c=malloc(sizeof(double)*m*n);
    if(c==NULL){perror("malloc c"); exit(1);}
    for(i=0; i<m*n; i++) c[i]=i;

    //dgemm(order,trans_a,trans_b,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc)
    // c[0..m][0..n] a[0..m][0..k] b[0..k][0..n]
    // lda=k ldb=n ldc=n
    t1=clock();
    for(i=0; i<10; i++)cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
	    m,n,k,alpha,a,k,b,n,beta,c,n);
    t2=clock();
    for(i=0; i<10; i++)my_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
	    m,n,k,alpha,a,k,b,n,beta,c,n);
    t3=clock();
    fprintf(logfile,"blas dgemm NN: %g\n", 10.0*n*m*k*CLOCKS_PER_SEC/(t2-t1));
    fprintf(logfile,"my   dgemm NN: %g\n", 10.0*n*m*k*CLOCKS_PER_SEC/(t3-t2));
    fprintf(logfile,"ratio        : %g\n", (t2-t1)/(double)(t3-t2));

    // transpose b
    // c[0..m][0..n] a[0..m][0..k] b[0..n][0..k]
    // lda=k ldb=k ldc=n
    t1=clock();
    for(i=0; i<10; i++)cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
	    m,n,k,alpha,a,k,b,k,beta,c,n);
    t2=clock();
    for(i=0; i<10; i++)my_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
	    m,n,k,alpha,a,k,b,k,beta,c,n);
    t3=clock();
    fprintf(logfile,"blas dgemm NT: %g\n", 10.0*n*m*k*CLOCKS_PER_SEC/(t2-t1));
    fprintf(logfile,"my   dgemm NT: %g\n", 10.0*n*m*k*CLOCKS_PER_SEC/(t3-t2));
    fprintf(logfile,"ratio        : %g\n", (t2-t1)/(double)(t3-t2));

    // transpose a
    // c[0..m][0..n] a[0..k][0..m] b[0..k][0..n]
    // lda=m ldb=n ldc=n
    t1=clock();
    for(i=0; i<10; i++)cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
	    m,n,k,alpha,a,m,b,n,beta,c,n);
    t2=clock();
    for(i=0; i<10; i++)my_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
	    m,n,k,alpha,a,m,b,n,beta,c,n);
    t3=clock();
    fprintf(logfile,"blas dgemm TN: %g\n", 10.0*n*m*k*CLOCKS_PER_SEC/(t2-t1));
    fprintf(logfile,"my   dgemm TN: %g\n", 10.0*n*m*k*CLOCKS_PER_SEC/(t3-t2));
    fprintf(logfile,"ratio        : %g\n", (t2-t1)/(double)(t3-t2));

    // transpose a,b
    // c[0..m][0..n] a[0..k][0..m] b[0..n][0..k]
    // lda=m ldb=k ldc=n
    t1=clock();
    for(i=0; i<10; i++)cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,
	    m,n,k,alpha,a,m,b,k,beta,c,n);
    t2=clock();
    for(i=0; i<10; i++)my_dgemm(CblasRowMajor,CblasTrans,CblasTrans,
	    m,n,k,alpha,a,m,b,k,beta,c,n);
    t3=clock();
    fprintf(logfile,"blas dgemm TT: %g\n", 10.0*n*m*k*CLOCKS_PER_SEC/(t2-t1));
    fprintf(logfile,"my   dgemm TT: %g\n", 10.0*n*m*k*CLOCKS_PER_SEC/(t3-t2));
    fprintf(logfile,"ratio        : %g\n", (t2-t1)/(double)(t3-t2));

    free(a);
    free(b);
    free(c);
    fclose(logfile);
    return 0;
}
