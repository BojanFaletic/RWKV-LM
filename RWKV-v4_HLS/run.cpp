#include <iostream>

#include "tensor.hpp"
#include "linear.hpp"

void test_vector(){
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

int main(){
    test_vector();


    return 0;

}