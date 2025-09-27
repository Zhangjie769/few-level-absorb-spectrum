// src/fft_backend.h
#pragma once
#include <vector>
#include <complex>
#include <cstring>
#include <fftw3.h>

namespace fft {

using cd = std::complex<double>;

struct Plan {
  fftw_plan fwd{}, inv{};
  std::vector<cd> inbuf, outbuf;
  size_t n{};
};

inline Plan make_plan(size_t n) {
  Plan p; p.n = n;
  p.inbuf.resize(n);
  p.outbuf.resize(n);
  p.fwd = fftw_plan_dft_1d(int(n),
      reinterpret_cast<fftw_complex*>(p.inbuf.data()),
      reinterpret_cast<fftw_complex*>(p.outbuf.data()),
      FFTW_FORWARD, FFTW_ESTIMATE);
  p.inv = fftw_plan_dft_1d(int(n),
      reinterpret_cast<fftw_complex*>(p.inbuf.data()),
      reinterpret_cast<fftw_complex*>(p.outbuf.data()),
      FFTW_BACKWARD, FFTW_ESTIMATE);
  return p;
}

inline void destroy(Plan& p){
  if(p.fwd) fftw_destroy_plan(p.fwd);
  if(p.inv) fftw_destroy_plan(p.inv);
  p.inbuf.clear(); p.outbuf.clear();
}

inline void forward(Plan& p, const std::vector<cd>& in, std::vector<cd>& out){
  const size_t n = p.n;
  out.resize(n);
  std::memcpy(p.inbuf.data(), in.data(), n*sizeof(cd));
  fftw_execute(p.fwd);
  std::memcpy(out.data(), p.outbuf.data(), n*sizeof(cd));
}

inline void backward(Plan& p, const std::vector<cd>& in, std::vector<cd>& out){
  const size_t n = p.n;
  out.resize(n);
  std::memcpy(p.inbuf.data(), in.data(), n*sizeof(cd));
  fftw_execute(p.inv);
  std::memcpy(out.data(), p.outbuf.data(), n*sizeof(cd));
}

} // namespace fft