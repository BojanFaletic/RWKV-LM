#pragma once
#include <array>
#include <algorithm>
#include <ostream>
#include <random>
#include <iostream>

using uint = unsigned int;

template <typename T, uint len>
class Tensor1d
{
private:
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
        std::for_each_n(&tensor[0], len, [](T &x)
                        { x = std::rand(); });
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

    T &operator[](int const idx)
    {
        return tensor[idx];
    }

    T operator[] (int const idx) const
    {
        return tensor[idx];
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

    Tensor1d<T, w> &operator[](int const idx)
    {
        return tensor[idx];
    }

    Tensor1d<T, w> operator[] (int const idx) const
    {
        return tensor[idx];
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

    Tensor2d<T, w, h> &operator[] (int const idx) const
    {
        return tensor[idx];
    }
};