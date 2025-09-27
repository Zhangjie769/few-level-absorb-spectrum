#pragma once
#include <array>
#include <complex>
#include <vector>

// 4-level: a'(t) = -i * E(t) * H(t) * a(t)
// 这里 H(t) 的作用等价于你 MATLAB 里那组 exp(-i dE t) * d * a。
// 为了和你脚本一致：每一步都用 exp(-i dE * t) * d * a 来构造“导数”。

using cd = std::complex<double>;
using vec4 = std::array<cd,4>;

// 乘以 exp(-i * dE_row[:] * t) .* d(row,:) * a
inline cd row_op(const std::array<double,4>& dErow, const std::array<double,4>& drow,
                 const vec4& a, double t)
{
    cd sum = 0.0;
    for(int j=0;j<4;++j){
        double phase = - dErow[j] * t;
        cd ph = cd(std::cos(phase), std::sin(phase)); // exp(i*phase)
        sum += ph * (drow[j] * a[j]);
    }
    return sum;
}

inline void rhs_da(vec4& out, const std::array<std::array<double,4>,4>& dE,
                   const std::array<std::array<double,4>,4>& d,
                   const vec4& a, double t, double E_t)
{
    // k(row) = -i * E(t) * [exp(-i dE(row,:) t) .* d(row,:) * a]
    const cd minus_i(0.0, -1.0);
    for(int r=0;r<4;++r){
        std::array<double,4> dErow{}, drow{};
        for(int j=0;j<4;++j){ dErow[j]=dE[r][j]; drow[j]=d[r][j]; }
        cd tmp = row_op(dErow, drow, a, t);
        out[r] = minus_i * E_t * tmp;
    }
}

inline void rk4_step(vec4& a, double t, double dt,
                     const std::array<std::array<double,4>,4>& dE,
                     const std::array<std::array<double,4>,4>& d,
                     double E_t, double E_t_half, double E_t_next)
{
    vec4 k1{}, k2{}, k3{}, k4{};
    vec4 a1=a, a2, a3;

    rhs_da(k1, dE, d, a, t,           E_t);
    for(int i=0;i<4;++i) a2[i] = a[i] + 0.5*dt*k1[i];

    rhs_da(k2, dE, d, a2, t+0.5*dt,   E_t_half);
    for(int i=0;i<4;++i) a3[i] = a[i] + 0.5*dt*k2[i];

    rhs_da(k3, dE, d, a3, t+0.5*dt,   E_t_half);
    vec4 a4;
    for(int i=0;i<4;++i) a4[i] = a[i] + dt*k3[i];

    rhs_da(k4, dE, d, a4, t+dt,       E_t_next);

    for(int i=0;i<4;++i) a[i] += (dt/6.0)*(k1[i] + cd(2)*k2[i] + cd(2)*k3[i] + k4[i]);
}