"""
VIST Compute Core - High Precision Spatial Frequency Response (SFR) Module
-------------------------------------------------------------------------
This module provides a pure implementation of the Slanted-Edge SFR algorithm 
as defined in ISO 12233. It focuses on converting raw image radiance data 
into the Modulation Transfer Function (MTF) to quantify the sharpness and 
resolving power of optical systems.

本模块是 VIST 系统的底层算法组件，专注于斜边空间频率响应 (SFR) 的高精度计算。
代码逻辑严格对应 ISO 12233 标准中定义的物理与数学变换过程。

功能概述 (Functionality Overview):
-------------------------------
1. 亚像素边缘寻迹：基于一阶导数重心法 (Centroid Method) 实现亚像素级的边缘定位。
2. 几何线性拟合：通过最小二乘法精确估计斜边角度，用于后续的投影校正。
3. 信号重采样 (Binning)：通过 4 倍超采样技术打破像素网格的采样频率限制。
4. 传递函数转换：实现从 ESF (边缘扩散) 到 LSF (线扩散) 再到 MTF (调制度) 的转换。
5. 指标量化：自动插值获取 MTF50 (空间频率对比度下降至 50% 时的频率值)。

输出指标说明 (Output Metrics):
----------------------------
- MTF50 (主指标): 调制传递函数下降到 50% 时的空间频率（Cycles/Pixel）。
  这是衡量镜头/传感器锐度常用的指标，反映了视觉感知的清晰度。
- MTF Curve: 完整的频率响应曲线，展示了从低频到高频的对比度保持能力。
- Nyquist Frequency (0.5 cyc/px): 系统采样的物理极限频率。

算法流程 (Standard Mathematical Pipeline):
----------------------------------------
[Raw ROI] -> [Centroid Extraction] -> [Linear Fit] -> [ESF Projection] 
-> [LSF Differentiation] -> [Windowing] -> [FFT] -> [Normalized MTF]

注意事项：
- 输入图像必须为单通道灰度图 (Single-channel Grayscale)。
- 边缘斜率建议在 2 到 10 度之间，以获得最佳的超采样效果。


作者: Zhang Lei
日期: 2026-02-18
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

class ISO12233_SFR:
    """
    ISO 12233 空间频率响应 (SFR) 计算核心类。
    """
    def __init__(self, roi_image):
        """
        初始化 SFR 计算器
        :param roi_image: 输入的灰度图像 (单通道 numpy array)
        注意：图像应该是“黑白斜边”，且边缘接近垂直（如果是水平边，请先旋转90度）
        """
        if roi_image is None:
            raise ValueError("SFR Error: Input ROI is None.")
        if len(roi_image.shape) != 2:
            raise ValueError("SFR Error: Grayscale image expected (2D array).")
    
        self.roi = roi_image
        self.height, self.width = roi_image.shape

    def _find_edge_centroids(self):
        """
        步骤 1: 使用重心法找到每一行的边缘位置
        利用一阶中心差分寻找每一行的梯度峰值，并计算其加权重心。
        """
        # 使用 [-0.5, 0, 0.5] 卷积核或者简单的 np.gradient
        grads = np.gradient(self.roi.astype(float), axis=1)
        
        centroids = []
        valid_rows = []
        
        for y in range(self.height):
            row_grad = grads[y, :]
            # 找到梯度最大的位置作为粗略中心
            peak_idx = np.argmax(np.abs(row_grad))
            
            # 在峰值附近开一个窗口计算重心 (Centroid)
            # 窗口大小通常取 5-7 像素
            win_radius = 3
            start = max(0, peak_idx - win_radius)
            end = min(self.width, peak_idx + win_radius + 1)
            
            mass = np.sum(np.abs(row_grad[start:end]))
            if mass == 0: continue # 避免除以零
            
            # 重心公式: sum(x * weight) / sum(weight)
            indices = np.arange(start, end)
            centroid = np.sum(indices * np.abs(row_grad[start:end])) / mass
            
            centroids.append(centroid)
            valid_rows.append(y)
            
        return np.array(valid_rows), np.array(centroids)

    def _fit_line(self, rows, centroids):
        """
        步骤 2: 对边缘点进行线性拟合 x = my + c
        """
        if len(rows) < 5:
            return None, None # 有效行数太少
            
        # 使用 numpy 的多项式拟合 (1次多项式即直线)
        # fit: x = p[0]*y + p[1]
        # 最小二乘线性回归
        p = np.polyfit(rows, centroids, 1)
        slope = p[0]  # 斜率 (dx/dy)
        intercept = p[1]
        return slope, intercept

    def calculate_mtf(self, oversampling=4):
        """
        核心计算流程：ESF -> LSF -> FFT -> MTF。
        :param oversampling: 超采样率，默认为 4，对应 ISO 12233 推荐值。
        :return: (frequencies, mtf_curve, mtf50_value)
        """
        # 1. 寻找边缘
        rows, centroids = self._find_edge_centroids()
        if len(rows) < 10:
            return None, None, 0.0
            
        # 2. 拟合直线
        slope, intercept = self._fit_line(rows, centroids)
        if slope is None:
            return None, None, 0.0
            
        # 3. 构造 ESF (Edge Spread Function) 
        # 修正斜率带来的投影误差 factor = cos(atan(slope))
        cos_theta = 1.0 / np.sqrt(1 + slope**2)

        # 装箱 (Binning) 过程，将 2D 图像信息压缩至 1D 超采样序列
        # 将每个像素投影到垂直于边缘的轴上
        esf_bins = np.zeros(self.width * oversampling * 2) # 预留足够的桶
        esf_counts = np.zeros(self.width * oversampling * 2)
        

        center_bin = len(esf_bins) // 2
        
        for y in range(self.height):
            for x in range(self.width):
                # 计算该像素中心到拟合直线的水平距离
                # 拟合直线 x' = slope * y + intercept
                # 距离 = x - x'
                dist_x = x - (slope * y + intercept)
                
                # 投影距离
                dist_projected = dist_x * cos_theta
                
                # 放入对应的 Bin
                bin_idx = int(dist_projected * oversampling) + center_bin
                if 0 <= bin_idx < len(esf_bins):
                    esf_bins[bin_idx] += self.roi[y, x]
                    esf_counts[bin_idx] += 1
                    
        # 4. 整理 ESF 曲线
        # 去除空桶，生成平滑的 ESF 曲线。
        valid_indices = esf_counts > 0
        if np.sum(valid_indices) < 10:
             return None, None, 0.0

        esf_curve = esf_bins[valid_indices] / esf_counts[valid_indices]
        
        # 5. 计算 LSF (Line Spread Function)
        # 对 ESF 曲线求导，得到线扩散函数（LSF）
        lsf_curve = np.gradient(esf_curve)
        
        # 6. 加窗 (Hamming Window)
        # 必须进行信号加窗处理，否则 FFT 会在不连续处产生严重的频谱泄漏。
        window = np.hanning(len(lsf_curve))
        lsf_windowed = lsf_curve * window
        
        # 7. FFT 变换得到 MTF
        mtf_complex = np.fft.fft(lsf_windowed, n=2048) # 补零到 2048 点提高频域分辨率
        mtf_real = np.abs(mtf_complex[:1024]) # 取正频率的幅值
        
        # 归一化 (直流分量 DC = 1)
        if mtf_real[0] == 0: return None, None, 0.0
        mtf_normalized = mtf_real / mtf_real[0]
        
        # 频率轴生成 (Cycles/Pixel)  
        freqs = np.linspace(0, oversampling, len(mtf_normalized))
        
        # 8. 计算 MTF50
        # Nyquist 频率是 0.5，对应索引 len/2
        # 只看 0 到 0.5 cycles/pixel 的范围
        valid_range = freqs <= 0.5
        target_freqs = freqs[valid_range]
        target_mtf = mtf_normalized[valid_range]
        
        try:
            # 寻找对比度下降至 0.5 时的空间频率
            mtf50 = np.interp(0.5, target_mtf[::-1], target_freqs[::-1])
        except:
            mtf50 = 0.0
            
        return target_freqs, target_mtf, mtf50

# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 读取一张包含垂直边缘的 ROI (必须是灰度)
    img = cv2.imread(r".\temp\edge_test.png", 0) 
  
    if img is not None:
        # 2. 如果是水平边，先转置: img = img.T
        
        # 3. 初始化计算器
        sfr_calc = ISO12233_SFR(img)
        
        # 4. 计算
        freqs, mtf, val = sfr_calc.calculate_mtf()
        print(f"MTF50: {val:.4f} cycles/pixel")
        
        # 5. 简单画图看看

        plt.figure(figsize=(8, 4))
        plt.plot(freqs, mtf, label='MTF Curve', lw=2)
        plt.axhline(0.5, color='r', linestyle='--', label='50% Contrast Threshold')
        plt.title("VIST-Compute-Core: Internal SFR Kernel Test")
        plt.xlabel("Spatial Frequency (Cycles/Pixel)")
        plt.ylabel("Modulation (Contrast)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()