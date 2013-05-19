#include<stddef.h>
#include<cblas.h>
#include<math.h>
#include"myblas.h"

void my_daxpy(size_t n, double alpha, const double* x, size_t incx,
	double* y, size_t incy)
{
    size_t i;
    if(alpha==0.0)return;
    for(i=0; i<n; i++){
	y[i*incy]+=alpha*x[i*incx];
    }
}
double my_ddot(size_t n, const double* dx, size_t incx,
	const double* dy, size_t incy)
{
    double sum=0.0;
    size_t i;
    for(i=0; i<n; i++){
	sum+=dx[i*incx]*dy[i*incy];
    }
    return sum;
}
void my_dscal(size_t n, double alpha, double* x, size_t incx)
{
    size_t i;
    if(alpha==1.0) return;
    for(i=0; i<n; i++){
	x[i*incx]*=alpha;
    }
}
size_t my_idamax(size_t n, const double *x, size_t incx)
{
    double xmax;
    size_t imax, i;
    if(n==0){
	return 0;
    }
    imax=0;
    xmax=fabs(x[0]);
    for(i=1; i+1<n; i+=2){
	double xi=fabs(x[i*incx]);
	double xi1=fabs(x[(i+1)*incx]);
	if(xi1>xi){
	    if(xi1>xmax){
		xmax=xi1;
		imax=i+1;
	    }
	}else{
	    if(xi>xmax){
		xmax=xi;
		imax=i;
	    }
	}
    }
    for(; i<n; i++){
	double xi=fabs(x[i*incx]);
	if(xi>xmax){
	    xmax=xi;
	    imax=i;
	}
    }
    return imax;
}
