#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <complex>

template<typename T>
void save_vector_csv(const std::string& path, const std::vector<T>& v, const std::string& header="") {
    std::ofstream ofs(path);
    if(!ofs) return;
    if(!header.empty()) ofs << "# " << header << "\n";
    for(size_t i=0;i<v.size();++i) ofs << v[i] << "\n";
}

template<typename T>
void save_vector2_csv(const std::string& path, const std::vector<T>& x, const std::vector<T>& y, const std::string& header="x,y") {
    std::ofstream ofs(path);
    if(!ofs) return;
    ofs << "# " << header << "\n";
    for(size_t i=0;i<x.size();++i) ofs << x[i] << "," << y[i] << "\n";
}

template<typename T>
void save_matrix_csv(const std::string& path, const std::vector<std::vector<T>>& M, const std::string& header="") {
    std::ofstream ofs(path);
    if(!ofs) return;
    if(!header.empty()) ofs << "# " << header << "\n";
    for(const auto& row : M){
        for(size_t j=0;j<row.size();++j){
            ofs << row[j] << (j+1==row.size()?'\n':',');
        }
    }
}