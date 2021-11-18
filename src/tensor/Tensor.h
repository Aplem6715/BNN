#ifndef TENSOR_H_
#define TENSOR_H_

template <typename T, int SizeX, int SizeY>
struct Tensor2D
{
    T data[SizeY][SizeX];

    T *operator[](int i)
    {
        return &data[i];
    }
};

template <typename T, int SizeX, int SizeY, int SizeZ>
struct Tensor3D
{
    static constexpr int shape[3] = {SizeX, SizeY, SizeZ};

    Tensor2D<T, SizeX, SizeY> data[SizeZ];

    Tensor2D<T, SizeX, SizeY> *operator[](int i)
    {
        return data[i];
    }
};

template <int SizeX, int SizeY, int SizeZ>
using RealTensor3D = Tensor3D<double, SizeX, SizeY, SizeZ>;

template <int SizeX, int SizeY>
using RealTensor2D = Tensor2D<double, SizeX, SizeY>;

#endif