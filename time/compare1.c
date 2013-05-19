#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<cblas.h>
#include"myblas.h"

int main(int argc, char *argv[])
{
    size_t n=20000000;
    double *x, *y;
    double alpha=2.0;
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
    fprintf(logfile,"blas daxpy: %g\n", 10.0*n*CLOCKS_PER_SEC/(t2-t1));
    fprintf(logfile,"my   daxpy: %g\n", 10.0*n*CLOCKS_PER_SEC/(t3-t2));
    fprintf(logfile,"ratio     : %g\n", (t2-t1)/(double)(t3-t2));

    t1=clock();
    for(i=0; i<10; i++){volatile double z=cblas_ddot(n,x,1,y,1);}
    t2=clock();
    for(i=0; i<10; i++){volatile double z=my_ddot(n,x,1,y,1);}
    t3=clock();
    fprintf(logfile,"blas ddot: %g\n", 10.0*n*CLOCKS_PER_SEC/(t2-t1));
    fprintf(logfile,"my   ddot: %g\n", 10.0*n*CLOCKS_PER_SEC/(t3-t2));
    fprintf(logfile,"ratio    : %g\n", (t2-t1)/(double)(t3-t2));

    t1=clock();
    for(i=0; i<10; i++)cblas_dscal(n,alpha,x,1);
    t2=clock();
    for(i=0; i<10; i++)my_dscal(n,alpha,x,1);
    t3=clock();
    fprintf(logfile,"blas dscal: %g\n", 10.0*n*CLOCKS_PER_SEC/(t2-t1));
    fprintf(logfile,"my   dscal: %g\n", 10.0*n*CLOCKS_PER_SEC/(t3-t2));
    fprintf(logfile,"ratio     : %g\n", (t2-t1)/(double)(t3-t2));

    t1=clock();
    for(i=0; i<10; i++){volatile int j=cblas_idamax(n,x,1);}
    t2=clock();
    for(i=0; i<10; i++){volatile size_t j=my_idamax(n,x,1);}
    t3=clock();
    fprintf(logfile,"blas idamax: %g\n", 10.0*n*CLOCKS_PER_SEC/(t2-t1));
    fprintf(logfile,"my   idamax: %g\n", 10.0*n*CLOCKS_PER_SEC/(t3-t2));
    fprintf(logfile,"ratio      : %g\n", (t2-t1)/(double)(t3-t2));

    free(x);
    free(y);
    fclose(logfile);
    return 0;
}
