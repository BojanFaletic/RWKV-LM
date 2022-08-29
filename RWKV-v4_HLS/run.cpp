#include <iostream>

#include "tensor.hpp"
#include "layers.hpp"

void test_vector()
{
    auto T2 = Tensor1d<short, 16>::rand();
    auto T1 = Tensor2d<short, 16, 2>::zeros();
    auto T0 = Tensor3d<short, 16, 2, 1>::zeros();

    std::cout << T2 << '\n';
    std::cout << T2[1] << '\n';

    T2[1] = 7;

    std::cout << T2[1] << '\n';

    std::cout << T1 << '\n';

    T1[0][0] = 11;
    std::cout << T1[0] << '\n';

    std::cout << "Hello world \n";

    std::cout << T0 << '\n';


    auto T4 = Tensor1d<float, 4>::rand();
    std::cout << T4 << '\n';
    std::cout << "Mean: " << T4.mean() << '\n';
    std::cout << "Var: " << T4.var() << '\n';
    std::cout << "sqrt: " << T4.sqrt() << '\n';


}

void test_linear()
{
    Linear L0 = Linear<short, 2, 3>();

    Tensor2d T0 = Tensor2d<short, 2, 3>::ones();

    auto k = L0(T0);

    std::cout << k;
}

void test_ln(){
    Tensor1d T0 = Tensor1d<float, 8>::rand();
    LayerNorm ln = LayerNorm<float, 8>();

    std::cout << T0 << '\n';
    std::cout << ln(T0) << '\n';

    std::cout << "mean:" << T0.mean() << '\n';
    std::cout << "var:" << T0.var() << '\n';


    Tensor2d T1 = Tensor2d<float, 1, 8>::rand();
    std::cout << ln(T1) << '\n';


}


int main()
{
    //test_vector();
    //test_linear();
    test_ln();

    return 0;
}