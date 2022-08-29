#pragma once
#include <array>
#include <algorithm>
#include <ostream>
#include <random>
#include <iostream>
#include <limits.h>

using uint = unsigned int;

template <typename T, uint len>
class Tensor1d;


namespace std
{
    template <typename _InputIterator, typename _Size, typename _Function>
    _GLIBCXX20_CONSTEXPR
        _InputIterator
        for_each_n(_InputIterator __first, _Size __n, _Function __f);


    template<typename T, uint n>
    Tensor1d<T, n> sqrt(Tensor1d<T, n> const &v){
        Tensor1d out{v};
        for (auto &el : out.tensor){
            el = std::sqrt(el);
        }
        return out;
    }
};

template <typename T, uint len>
class Tensor1d
{
public:
    std::array<T, len> tensor;

public:
    Tensor1d() = default;

    Tensor1d(const std::array<T, len> &init) : tensor{init}
    {
    }

    static Tensor1d<T, len> zeros()
    {
        std::array<T, len> tensor;
        std::for_each_n(&tensor[0], len, [](T &x)
                        { x = 0; });
        return Tensor1d(tensor);
    }

    static Tensor1d<T, len> ones()
    {
        std::array<T, len> tensor;
        std::for_each_n(&tensor[0], len, [](T &x)
                        { x = 1; });
        return Tensor1d(tensor);
    }

    static Tensor1d<T, len> rand()
    {
        std::array<T, len> tensor;

        // This could be better (use correct normal distribution)
        std::for_each_n(&tensor[0], len, [=](T &x)
                        { x = static_cast<T>(std::rand() % 1000) / 500; });
        return Tensor1d(tensor);
    }

    friend std::ostream &operator<<(std::ostream &os, Tensor1d const &ten)
    {
        os << "[ ";
        int i = 0;
        for (auto &el : ten.tensor)
        {
            os << el;
            os << " ";
            if (i++ > 8)
            {
                os << "... ";
                break;
            }
        }
        os << "]";
        return os;
    }

    Tensor1d operator+(T const i)
    {
        Tensor1d out{tensor};
        std::for_each_n(out.tensor.begin(), len, [&](T &x)
                        { x += i; });
        return out;
    }

    Tensor1d operator-(T const i) const
    {
        Tensor1d out{tensor};
        std::for_each_n(out.tensor.begin(), len, [&](T &x)
                        { x -= i; });
        return out;
    }

    Tensor1d operator/(T const i)
    {
        Tensor1d out{tensor};
        std::for_each_n(out.tensor.begin(), len, [&](T &x)
                        { x /= i; });
        return out;
    }

    Tensor1d operator*(T const i)
    {
        Tensor1d out{tensor};
        std::for_each_n(out.tensor.begin(), len, [&](T &x)
                        { x *= i; });
        return out;
    }

    Tensor1d operator*(Tensor1d const &ten) const
    {
        Tensor1d out{tensor};
        uint i=0;
        std::for_each_n(out.tensor.begin(), len, [&](T &x)
                        { x *= ten[i++]; });
        return out;
    }

    Tensor1d operator+(Tensor1d const &ten) const
    {
        Tensor1d out{tensor};
        uint i=0;
        std::for_each_n(out.tensor.begin(), len, [&](T &x)
                        { x += ten[i++]; });
        return out;
    }

    T &operator[](int const idx)
    {
        return tensor[idx];
    }

    T operator[](int const idx) const
    {
        return tensor[idx];
    }

    T mean() const
    {
        return std::accumulate(tensor.begin(), tensor.end(), 0.) / len;
    }

    T var() const
    {
        T _mean = mean();
        T acc = 0;
        std::for_each_n(tensor.begin(), len, [&](T x)
                        { acc += std::pow(x - _mean, 2); });
        return acc / len;
    }

    Tensor1d sqrt() const{
        Tensor1d ten = Tensor1d(tensor);
        std::for_each_n(ten.tensor.begin(), len, [&](T &x)
                        { x = std::sqrt(x); });
        return ten;
    }
};

template <typename T, uint h, uint w>
class Tensor2d
{
private:
    std::array<Tensor1d<T, w>, h> tensor;

public:
    std::array<uint, 2> shape{h, w};
    Tensor2d() = default;

    Tensor2d(const std::array<Tensor1d<T, w>, h> &init) : tensor{init}
    {
    }

    static Tensor2d zeros()
    {
        std::array<Tensor1d<T, w>, h> tensor;
        std::for_each_n(&tensor[0], h, [](auto &x)
                        { x = Tensor1d<T, w>::zeros(); });
        return Tensor2d(tensor);
    }

    static Tensor2d ones()
    {
        std::array<Tensor1d<T, w>, h> tensor;
        std::for_each_n(&tensor[0], h, [](auto &x)
                        { x = Tensor1d<T, w>::ones(); });
        return Tensor2d(tensor);
    }

    static Tensor2d rand()
    {
        std::array<Tensor1d<T, w>, h> tensor;
        std::for_each_n(&tensor[0], h, [](auto &x)
                        { x = Tensor1d<T, w>::rand(); });
        return Tensor2d(tensor);
    }

    friend std::ostream &operator<<(std::ostream &os, Tensor2d const &ten)
    {
        os << "[";
        int i = 0;
        for (auto &el : ten.tensor)
        {
            if (i++ != 0)
            {
                os << " ";
            }
            os << el;
            os << "\n";
            if (i > 4)
            {
                os << " ... ";
                break;
            }
        }
        os << "]";
        return os;
    }

    template<typename Tt>
    Tensor2d<T, h, w> operator-(Tt const i) const{
        Tensor2d out{tensor};

        // TODO: FIX this
        for (Tensor1d<T, w> &el : out.tensor){
            for (auto &ee : el.tensor){
                std::cout << ee;
                //ee = ee - i;
            }
        }
        return out;
    }

    template<typename Tt, uint len>
    Tensor2d<T, h, w> operator/(Tensor1d<Tt, len> const &ten) const{
        Tensor2d<T, h, w> out{tensor};

        for (auto &el : out.tensor){
            el = el / ten;
        }
        return out;
    }

    Tensor1d<T, w> &operator[](int const idx)
    {
        return tensor[idx];
    }

    Tensor1d<T, w> operator[](int const idx) const
    {
        return tensor[idx];
    }

    Tensor1d<T, w> mean() const
    {
        std::array<T, w> mean_values;
        for (uint i=0; i<w; i++){
            mean_values[i] = tensor[i].mean();
        }
        return Tensor1d<T, w>(mean_values);
    }

    Tensor1d<T, w> var() const
    {
        std::array<T, w> var_values;
        for (uint i=0; i<w; i++){
            var_values[i] = tensor[i].var();
        }
        return Tensor1d<T, w>(var_values);
    }
};

template <typename T, uint h, uint w, uint z>
class Tensor3d
{
private:
    std::array<Tensor2d<T, w, h>, z> tensor;

public:
    std::array<uint, 3> SH{h, w, z};
    Tensor3d() = default;

    Tensor3d(const std::array<Tensor2d<T, w, h>, z> &init) : tensor{init}
    {
    }

    static Tensor3d zeros()
    {
        std::array<Tensor2d<T, w, h>, z> tensor;
        std::for_each_n(&tensor[0], z, [](auto &x)
                        { x = Tensor2d<T, w, h>::zeros(); });
        return Tensor3d(tensor);
    }

    static Tensor3d ones()
    {
        std::array<Tensor2d<T, w, h>, z> tensor;
        std::for_each_n(&tensor[0], z, [](auto &x)
                        { x = Tensor2d<T, w, h>::ones(); });
        return Tensor3d(tensor);
    }

    static Tensor3d rand()
    {
        std::array<Tensor2d<T, w, h>, z> tensor;
        std::for_each_n(&tensor[0], z, [](auto &x)
                        { x = Tensor2d<T, w, h>::rand(); });
        return Tensor3d(tensor);
    }

    friend std::ostream &operator<<(std::ostream &os, Tensor3d const &ten)
    {
        os << "[";
        int i = 0;
        for (auto &el : ten.tensor)
        {
            if (i++ != 0)
            {
                os << " ";
            }
            os << el;
            os << "\n";
            if (i > 4)
            {
                os << " ... ";
                break;
            }
        }
        os << "]";
        return os;
    }

    Tensor2d<T, w, h> &operator[](int const idx)
    {
        return tensor[idx];
    }

    Tensor2d<T, w, h> &operator[](int const idx) const
    {
        return tensor[idx];
    }
};