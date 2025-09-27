#pragma once
#include <vector>
#include <complex>
#include <algorithm>

// in-place fftshift for complex arrays
template<typename T>
void fftshift(std::vector<std::complex<T>>& x) {
    const size_t N = x.size();
    const size_t k = N/2;
    std::rotate(x.begin(), x.begin()+k, x.end());
}

// inverse shift
template<typename T>
void ifftshift(std::vector<std::complex<T>>& x) {
    const size_t N = x.size();
    const size_t k = (N+1)/2;
    std::rotate(x.begin(), x.begin()+k, x.end());
}

// real-valued linear convolution (simple O(NM) version)
template<typename T>
std::vector<T> conv_real(const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> y(a.size() + b.size() - 1, T(0));
    for (size_t i=0;i<a.size();++i)
        for (size_t j=0;j<b.size();++j)
            y[i+j] += a[i]*b[j];
    return y;
}