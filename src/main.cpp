#include <bits/stdc++.h>
#include "fft_backend.h"
#include "fft_utils.h"
#include "io_utils.h"
#include "rk4_utils.h"

using namespace std;

// ========== 你的 main（已替换为统一 FFT 接口） ==========
int main(){
    // 常数
    const double c = 2.99792458e8;
    const double pi= 3.14159265358979323846;
    const double eps0 = 8.854187817e-12;
    const double h = 6.6260693e-34;
    const double hbar = h/(2.0*pi);
    const double e = 1.6021892e-19;
    const double me= 9.109534e-31;

    const double es  = e / std::sqrt(4.0*pi*eps0);
    const double tau = pow(hbar,3)/(me*pow(es,4));
    const double lau = 4.0*pi*eps0*hbar*hbar/(me*e*e);
    const double eau = pow(me*e*e/(4.0*pi*eps0*hbar),2);

    // 参数
    const int    k  = 1024*80;
    const double I1 = 1e12;
    const double I2 = 20e13;

    const double lambda1 = 2*pi*3/(65/27.211)*1e14*tau*1e-6;
    const double lambda2 = 800e-9;

    const double T1 = lambda1/c/tau;
    const double w1 = 2*pi/T1;
    const double T2 = lambda2/c/tau;
    const double w2 = 2*pi/T2;

    auto calcE = [&](double I, double lambda, double w){
        return sqrt(I)*eps0*lambda*me*c*1e8/e*w/tau*lau/27.2;
    };
    const double E1 = calcE(I1, lambda1, w1);
    const double E2 = calcE(I2, lambda2, w2);

    array<double,4> E = {0.0, 65.28/27.2, 63.73/27.2, 66.0/27.2};

    array<array<double,4>,4> dE{};
    for(int i=0;i<4;++i) for(int j=0;j<4;++j) dE[i][j] = E[j]-E[i];

    array<array<double,4>,4> d{};
    d[0][1]=0.02; d[1][0]=0.02;
    d[1][2]=2.7;  d[2][1]=2.7;

    // 时间/频率网格
    const double T = 120.0*T2;
    vector<double> t(k);
    const double dt = T/(k-1);
    for(int i=0;i<k;i++) t[i] = -T/2.0 + i*dt - dt/2.0;

    const double W = 2*pi/dt;
    const double dw = W/(k-1);
    vector<double> w(k);
    for(int i=0;i<k;i++) w[i]= i*dw;

    auto round_to_int = [](double x)->int { return (int) std::llround(x); };
    const int N1 = round_to_int(50/27.2/dw);
    const int N2 = round_to_int(80/27.2/dw);
    const int NF = (N2>=N1 ? (N2-N1+1) : 0);

    // delay/v
    const int m  = 160;
    vector<double> delay(m);
    for(int i=0;i<m;i++) delay[i] = -15.0*T2 + i*(30.0*T2/(m-1));

    const double dd = 15.0*T2/(m-1);
    const double V  = 2*pi/dd;
    const double dv = V/(m-1);
    vector<double> v(m); for(int i=0;i<m;i++) v[i]= i*dv;

    // 场
    const double Tuv  = 1e-18/tau;
    const double Tir  = 15000e-18/tau;
    vector<double> Euv(k), Eir(k);
    for(int n=0;n<k;n++){
        const double tn = t[n];
        Euv[n] = E1*exp(-2*log(2.0)*tn*tn/(Tuv*Tuv))*cos(w1*tn);
        Eir[n] = E2*cos(w2*tn)*exp(-2*log(2.0)*(tn/Tir)*(tn/Tir));
    }

    std::filesystem::create_directories("../out");

    // 统一 FFT 计划
  
    

    // 结果
    vector<vector<double>> P1(k, vector<double>(m, 0.0));
    vector<vector<double>> PP(NF, vector<double>(m, 0.0));
    vector<vector<double>> PF(NF, vector<double>(m, 0.0));

    // 窗
    vector<double> wel(k);
    {
        double max2t = *max_element(t.begin(), t.end())*2.0;
        for(int n=0;n<k;n++) wel[n] = pow(cos( (t[n])*pi/(max2t) ), 2.0);
    }
    #pragma omp parallel
{
  fft::Plan plan = fft::make_plan(k);
    // 并行外层（可去掉 OpenMP，也能跑）
    #pragma omp for schedule(dynamic,1)
    for(int s=0; s<m; ++s){
        vector<double> Efield(k), Efields(k);
        int shift = round_to_int(delay[s]/dt);
        for(int n=0;n<k;n++){
            int idx = ( (n - shift) % k + k ) % k;
            Efield[n] = Euv[n] + Eir[idx];
        }

        // Y = fftshift(fft(Efield))
        vector<complex<double>> Yin(k), Y, Y1, yout;
        for(int n=0;n<k;n++) Yin[n] = complex<double>(Efield[n], 0.0);
        fft::forward(plan, Yin, yout);
        Y = yout; //for(auto& z:Y) z /= double(k);
        fftshift(Y);

        // Y1 = Y .* exp(i*dt*w/2)
        Y1.resize(k);
        for(int n=0;n<k;n++){
            double ph = dt*w[n]/2.0;
            Y1[n] = Y[n]*complex<double>(cos(ph), sin(ph));
        }

        // Efields = imag(ifft(fftshift(Y1)))
        fftshift(Y1);
        fft::backward(plan, Y1, yout);
        for(int n=0;n<k;n++){
            complex<double> val = yout[n]/double(k);
            Efields[n] = imag(val);
        }

        // RK4
        vec4 a = { 1.0, 0.0, 0.0, 0.0 };
        vector<double> P(k, 0.0);
        array<array<double,4>,4> dloc = d;
        const double tmin = t.front();
        vector<double> t1(k); for(int n=0;n<k;n++) t1[n]=t[n]-tmin;

        for(int n=0;n<k-1;n++){
            rk4_step(a, t1[n], dt, dE, d, Efield[n], Efields[n], Efield[n+1]);

            array<complex<double>,4> temp{};
            for(int i=0;i<4;++i){
                double ph = -E[i]*t1[n];
                temp[i] = a[i] * complex<double>(cos(ph), sin(ph));
            }
            complex<double> Pn = 0.0;
            for(int i=0;i<4;++i){
                complex<double> rowSum = 0.0;
                for(int j=0;j<4;++j) rowSum += dloc[i][j]*temp[j];
                Pn += conj(temp[i]) * rowSum;
            }
            P[n] = Pn.real();
        }

        for(int n=0;n<k;n++) P1[n][s] = P[n]*wel[n];

        // Y0 = fft(fftshift(P1(:,s)))，截取 N1:N2
        for(int n=0;n<k;n++) Yin[n] = complex<double>(P1[n][s], 0.0);
        fftshift(Yin);
        fft::forward(plan, Yin, yout);
        vector<complex<double>> Y0 = yout;// for(auto& z:Y0) z/=double(k);
        for(int i=0;i<NF;i++){
            int idx = N1+i;
            if(idx>=0 && idx<k){
                PP[i][s] = imag(Y0[idx]);
                PF[i][s] = norm(Y0[idx]);
            }
        }

        #pragma omp critical
        fprintf(stderr, "done s=%d/%d\n", s+1, m);
    }
fft::destroy(plan);
}
    // 卷积 packet
    vector<double> ww(NF); for(int i=0;i<NF;i++) ww[i]= w[N1+i];
    vector<double> packet(NF);
    {
        const double inv_s = 1.0/(sqrt(2*pi)*0.8);
        double mid = (ww.front()+ww.back())/2.0;
        for(int i=0;i<NF;i++){
            double x = (ww[i] - (ww.front()/2.0) - (ww.back()/2.0))*27.2/0.3;
            packet[i] = inv_s * exp( - x*x );
        }
    }
    save_vector2_csv("../out/packet.csv", ww, packet, "w,packet");

    vector<vector<double>> PPpacket(NF + NF - 1, vector<double>(m, 0.0));
    for(int s=0;s<m;s++){
        vector<double> col(NF);
        for(int i=0;i<NF;i++) col[i] = pow(10.0, -PP[i][s]);
        auto conv = conv_real(col, packet);
        for(size_t i=0;i<conv.size();++i){
            double v = conv[i]; if(v<=0) v=1e-300;
            PPpacket[i][s] = - log10(v);
        }
    }

    // --- 居中裁剪为 "same" 大小（行数 = NF） ---
const int full_rows = (int)PPpacket.size(); // = NF + NF - 1
const int start = (full_rows - NF) / 2;     // 居中起点
vector<vector<double>> PPpacket_same(NF, vector<double>(m, 0.0));
for(int r=0; r<NF; ++r){
    for(int s=0; s<m; ++s){
        PPpacket_same[r][s] = PPpacket[start + r][s];
    }
}

    // AC-Stark shift 曲线
    vector<double> A(k); for(int n=0;n<k;n++) A[n] = E2 * exp(-2*log(2.0)* (t[n]/Tir)*(t[n]/Tir));
    vector<double> shift_curve(k);
    const double E23 = fabs(E[1]-E[2]);
    for(int n=0;n<k;n++){
        double x = 2.0 * d[1][2] * A[n]/w2;
        double J0 = std::cyl_bessel_j(0, x);
        double J1 = std::cyl_bessel_j(1, x);
        double term1 = (w2 - E23*J0);
        double term2 = E23*J1;
        shift_curve[n] = 27.2*0.5*sqrt(term1*term1 + term2*term2);
    }
    save_vector2_csv("./out/shift_curve.csv", t, shift_curve, "t,shift");

    // 保存
    save_matrix_csv("./out/P1.csv", P1, "rows=time, cols=delay");
    save_matrix_csv("./out/PP.csv", PP, "imag(FFT) segment N1:N2");
    save_matrix_csv("./out/PF.csv", PF, "|FFT|^2 segment N1:N2");
   //ave_matrix_csv("../out/PPpacket.csv", PPpacket, "-log10(conv(10.^(-PP), packet))");
    save_vector_csv("./out/t_axis.csv", t, "t");
    save_vector_csv("./out/w_axis.csv", w, "w");
    save_vector_csv("./out/delay_axis.csv", delay, "delay");
    save_vector_csv("./out/v_axis.csv", v, "v");
    save_matrix_csv("./out/PPpacket.csv", PPpacket_same, "PPpacket same-sized (centered)");
    {
        vector<double> ww_eV(NF); for(int i=0;i<NF;i++) ww_eV[i] = ww[i]*27.2;
        save_vector_csv("./out/ww_eV.csv", ww_eV, "w_segment_eV (N1:N2)");
    }

   
    fprintf(stderr, "Done. Data saved in ./out/\n");
    return 0;
}