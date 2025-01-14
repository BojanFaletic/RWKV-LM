#pragma once
#include "tensor.hpp"

using uint = unsigned int;

template <typename T, uint N, uint M>
class Linear
{
private:
    Tensor2d<T, N, M> weights;

public:
    Linear()
    {
        weights = Tensor2d<T, N, M>::ones();
    }

    template <typename tt, uint A, uint B>
    Tensor2d<T, A, N> operator()(Tensor2d<tt, A, B> const &data)
    {
        static_assert(std::is_same<tt, T>::value, "Not correct type");
        static_assert(A == N, "First dim not correct");
        static_assert(B == M, "Second dim not correct");

        constexpr uint width = A;
        constexpr uint height = N;
        constexpr uint dim = M;

        Tensor2d<T, width, height> tmp;

        for (uint h = 0; h < height; h++)
        {
            for (uint w = 0; w < width; w++)
            {
                T acc = 0;
                for (uint d = 0; d < dim; d++)
                {
                    acc += data[h][d] * weights[w][d];
                }
                tmp[w][h] = acc;
            }
        }
        return tmp;
    }
};

template <typename T, uint N>
class LayerNorm
{
private:
    Tensor1d<T, N> weights;
    Tensor1d<T, N> bias;
    const float eps = 1e-5;

public:
    LayerNorm()
    {
        weights = Tensor1d<T, N>::ones();
        bias = Tensor1d<T, N>::zeros();
    }

#if 0
    template<typename tt, uint M>
    Tensor1d<T, N> operator()(Tensor1d<tt, M> const &data)
    {
        static_assert(std::is_same<tt, T>::value, "Not same type");
        static_assert(N == M, "Wrong dimensions");

        Tensor1d out = (data - data.mean()) / std::sqrt(data.var() + eps);
        out = out * weights + bias;
        return out;
    }
#endif

    template <typename tt, uint H, uint W>
    Tensor2d<tt, H, W> operator()(Tensor2d<tt, H, W> const &data)
    {
        Tensor2d out = (data - data.mean()) / std::sqrt(data.var() + eps);
        out = out * weights;// + bias;
        //return out;
        return {};
    }
};