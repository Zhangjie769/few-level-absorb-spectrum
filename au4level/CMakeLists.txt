cmake_minimum_required(VERSION 3.12)
project(au4level CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_OPENMP "Enable OpenMP" ON)
if(USE_OPENMP)
  find_package(OpenMP QUIET)
endif()

# 手动找 FFTW（系统默认路径）
find_path(FFTW3_INCLUDE_DIR fftw3.h /usr/include /usr/local/include)
find_library(FFTW3_LIB         NAMES fftw3         PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(FFTW3_THREADS_LIB NAMES fftw3_threads PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)

if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIB)
  message(FATAL_ERROR "FFTW3 not found. Install with: sudo apt install -y libfftw3-dev")
endif()

message(STATUS "FFTW include: ${FFTW3_INCLUDE_DIR}")
message(STATUS "FFTW lib:     ${FFTW3_LIB}")
message(STATUS "FFTW threads: ${FFTW3_THREADS_LIB}")

add_executable(au4level
  src/main.cpp
  src/fft_utils.h
  src/io_utils.h
  src/rk4_utils.h
  src/fft_backend.h
)

target_include_directories(au4level PRIVATE ${FFTW3_INCLUDE_DIR})
target_link_libraries(au4level PRIVATE ${FFTW3_LIB} m)
if(FFTW3_THREADS_LIB)
  target_link_libraries(au4level PRIVATE ${FFTW3_THREADS_LIB})
endif()
if(USE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(au4level PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_options(au4level PRIVATE -O3 -march=native)






# ---- 可选：一键生成图片 ----
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_custom_target(plots
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/plot_all.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating plots with Python/matplotlib"
  )
endif()