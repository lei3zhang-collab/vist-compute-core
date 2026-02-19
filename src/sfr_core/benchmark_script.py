"""
VIST Compute Core - SFR Algorithm Validation & Synthetic Target Generator
-------------------------------------------------------------------------
This module provides a robust benchmarking suite for the ISO 12233 Spatial 
Frequency Response (SFR) engine. It integrates a high-precision slanted-edge 
generator capable of simulating optical degradation through tunable Gaussian 
blur and additive white Gaussian noise (AWGN). The primary objective is to 
validate the accuracy of the SFR kernel by comparing measured MTF values 
against known simulation parameters in a controlled environment.

VIST-Compute-Core: ISO 12233 SFR 算法验证与合成标靶生成模块

本模块是 VIST 系统的算法验证层，专为斜边 SFR（空间频率响应）引擎提供基准测试。
它通过数学建模生成符合 ISO 12233 标准的虚拟斜边图像，用于闭环验证算法的可靠性。

功能概述 (Functionality Overview):
-------------------------------
1. 高精度边缘生成：支持自定义角度（如 5.7 度）的阶跃信号生成。
2. 光学退化模拟：通过高斯模糊（Sigma）模拟透镜的点扩散函数（PSF）效应。
3. 噪声环境建模：引入正态分布噪声，测试算法在不同信噪比（SNR）下的鲁棒性。
4. 自动化基准测试：自动执行图像生成、MTF 提取及关键频率点（如 0.25 cy/px）的性能分析。
5. 数据可视化：同步展示生成的 ROI 区域与对应的 MTF 响应曲线。

输出指标 (Key Metrics):
----------------------------
- MTF50: 调制传递函数下降至 50% 时的频率，是评价图像锐度的核心数值。
- MTF@0.25: 在 1/2 奈奎斯特频率处的对比度，用于评估中频细节表现。

作者: Zhang Lei
最后编辑时间: 2026-02-19
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
try:
    from sfr_core.sfr_iso12233 import ISO12233_SFR
except ImportError:
    from sfr_iso12233 import ISO12233_SFR

def generate_test_edge(filename=r".\temp\edge_test.png", width=256, height=256, 
                       angle_deg=5.7, blur_sigma=1.0, noise_level=0.05):
    """
    全功能斜边生成器：支持自定义角度、模糊度和噪声。
    此函数模拟了从物方场景到传感器数字输出的物理成像过程。
    """
    print(f"正在生成图像: {filename} (模糊度 Sigma={blur_sigma}, 角度={angle_deg}度)")
    
    # 1. 创建高精度画布：使用 float32 避免中间计算的量化误差
    img_float = np.zeros((height, width), dtype=np.float32)
    slope = np.tan(np.radians(angle_deg))
    
    # 2. 填充基础黑白阶跃信号 (0.2 ~ 0.8 避免像素饱和)
    # 根据 ISO 规范，边缘两侧应保留足够动态范围，避免 Clipping 影响重心计算
    for y in range(height):
        # 根据斜率计算每一行边缘的亚像素位置
        edge_x = slope * (y - height//2) + width//2
        mask = np.arange(width) < edge_x
        img_float[y, mask] = 0.2
        img_float[y, ~mask] = 0.8
        
    # 3. 【集成】施加高斯模糊模拟 MTF 下降
    # 高斯核的 Sigma 直接对应光学系统的模糊半径
    if blur_sigma > 0:
         # 根据 Sigma 自动计算核大小 (6*sigma 原则)，确保能量分布完整
        ksize = int(6 * blur_sigma + 1)
        if ksize % 2 == 0: ksize += 1
        img_float = cv2.GaussianBlur(img_float, (ksize, ksize), blur_sigma)
        
    # 4. 添加噪声：模拟传感器读出噪声与光子噪声
    noise = np.random.normal(0, noise_level, (height, width)).astype(np.float32)
    img_final = np.clip(img_float + noise, 0, 1)
    
    # 5. 数据持久化：转回 8-bit 并保存，供外部调试参考
    img_uint8 = (img_final * 255).astype(np.uint8)
    cv2.imwrite(filename, img_uint8)
    return img_uint8

# --- 自动化测试与验证 ---
def run_benchmark():
    """
    运行算法基准测试：生成图像 -> 调用 SFR 内核 -> 分析指标。
    """
    # 设定目标模糊因子，用于验证算法对 MTF 下降的敏感度
    target_sigma = 1.2
    # 生成特定尺寸的 ROI (40x90)
    img = generate_test_edge(blur_sigma=target_sigma,width=40,height=90)
    
    # 实例化算法内核并执行计算
    sfr = ISO12233_SFR(img)
    freqs, mtf, mtf50 = sfr.calculate_mtf()
    
    # 提取关键频率点指标：利用线性插值获取 0.25 cy/pixel 处的响应值
    mtf_at_025 = np.interp(0.25, freqs, mtf)
    
    print("\n" + "="*40)
    print(f"仿真结果报告 (Sigma={target_sigma})")
    print(f"MTF50 频率值: {mtf50:.4f} cy/px")
    print(f"关键指标 MTF@0.25: {mtf_at_025:.4f}")
    print("="*40)

    # 图形化结果对比：验证频域响应与空域图像的一致性
    plt.figure(figsize=(12, 5))
    
    # 左侧：展示生成的原始 ROI 图像
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Generated ROI (Sigma={target_sigma})")
    
    # 右侧：绘制 MTF 响应曲线
    plt.subplot(1, 2, 2)
    plt.plot(freqs, mtf, 'b-', linewidth=2, label='MTF Curve')
    plt.axvline(0.25, color='r', linestyle='--', label='Target Freq 0.25')
    plt.scatter([0.25], [mtf_at_025], color='red', s=100)
    # 标注关键点数值
    plt.text(0.26, mtf_at_025 + 0.05, f'Value: {mtf_at_025:.3f}', color='red', fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.xlim(0, 0.5)  # 限制在奈奎斯特频率以内
    plt.xlabel("Frequency (cy/px)")
    plt.ylabel("Modulation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
        run_benchmark()
