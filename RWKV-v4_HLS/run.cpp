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
}

void test_linear()
{
    Linear L0 = Linear<short, 2, 3>();

    Tensor2d T0 = Tensor2d<short, 2, 3>::ones();

    auto k = L0(T0);

    std::cout << k;
}

int main()
{
    test_linear();

    return 0;
}