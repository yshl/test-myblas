#include<assert.h>
#include<stdlib.h>
#include<cblas.h>
#include"myblas.h"

void test_daxpy()
{
    const size_t n=5;
    double x[5]={1,2,3,4,5};
    double y[5]={6,7,8,9,10};
    double z[5]={6,7,8,9,10};
    size_t i; 
    my_daxpy(n,2.0,x,1,y,1);
    cblas_daxpy(n,2.0,x,1,z,1);
    for(i=0; i<n; i++){
	assert(y[i]==z[i]);
    }
}

void test_ddot()
{
    const size_t n=5;
    double x[5]={1,2,3,4,5};
    double y[5]={6,7,8,9,10};
    assert(my_ddot(n,x,1,y,1)==cblas_ddot(n,x,1,y,1));
}

void test_dscal()
{
    const size_t n=5;
    double x[5]={1,2,3,4,5};
    double y[5]={1,2,3,4,5};
    size_t i;

    my_dscal(n,2.0,x,1);
    my_dscal(n,2.0,y,1);
    for(i=0; i<n; i++){
	assert(x[i]==y[i]);
    }
}

void test_idamax()
{
    const size_t n=5;
    double x[5]={3,1,5,2,4};
    assert(my_idamax(n,x,1)==cblas_idamax(n,x,1));
}

int main()
{
    test_daxpy();
    test_ddot();
    test_dscal();
    test_idamax();
    return 0;
}
