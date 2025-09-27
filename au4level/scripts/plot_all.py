#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT, exist_ok=True)

def load_csv(path):
    return np.loadtxt(path, delimiter=",", comments="#")

# 读数据（你的 C++ 程序已输出这些）
P1         = load_csv(os.path.join(OUT, "P1.csv"))           # k x m
PP         = load_csv(os.path.join(OUT, "PP.csv"))           # NF x m
PF         = load_csv(os.path.join(OUT, "PF.csv"))           # NF x m
PPpacket   = load_csv(os.path.join(OUT, "PPpacket.csv"))     # (NF+NF-1) x m
t_axis     = load_csv(os.path.join(OUT, "t_axis.csv"))
w_axis     = load_csv(os.path.join(OUT, "w_axis.csv"))
delay_axis = load_csv(os.path.join(OUT, "delay_axis.csv"))
ww_eV      = load_csv(os.path.join(OUT, "ww_eV.csv"))
shift      = load_csv(os.path.join(OUT, "shift_curve.csv"))

# 方便起见，构造 extent
# delay 归一化单位：/T2（你的 C++ 没输出 T2，这里用相对 delay 的线性轴即可）
delay_min, delay_max = delay_axis[0], delay_axis[-1]
e_min, e_max = ww_eV[0], ww_eV[-1]

# ---------- 图1：PPpacket（推荐主图） ----------
plt.figure(figsize=(8,5))
rows = min(PPpacket.shape[0], ww_eV.shape[0])  # 对齐显示高度
plt.imshow(PPpacket[:rows,:], origin="lower", aspect="auto",
           extent=[delay_min, delay_max, e_min, ww_eV[rows-1]],
           cmap="jet",
           vmin=-1.3133, vmax=-1.31
           )
plt.colorbar(label="-log10(power)")
plt.xlabel("Delay (atomic time units)")
plt.ylabel("Energy (eV)")
plt.title("PPpacket")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "PPpacket.png"), dpi=200)
plt.close()

# ---------- 图2：PP（imag FFT segment） ----------
plt.figure(figsize=(8,5))
plt.imshow(PP, origin="lower", aspect="auto",
           extent=[delay_min, delay_max, e_min, e_max],
           cmap="jet")

plt.colorbar(label="Imag(FFT)",
           )
plt.xlabel("Delay")
plt.ylabel("Energy (eV)")
plt.title("PP (Imag FFT segment N1:N2)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "PP.png"), dpi=200)
plt.close()

# ---------- 图3：AC-Stark shift 曲线 ----------
plt.figure(figsize=(7,4))
plt.plot(shift[:,0], shift[:,1])
plt.xlabel("t")
plt.ylabel("Shift (eV)")
plt.title("AC-Stark shift curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "shift_curve.png"), dpi=200)
plt.close()

print("Saved figures to", OUT)