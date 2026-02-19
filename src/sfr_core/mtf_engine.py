"""
VIST Compute Core - Multi-directional SFR Analysis & ROI Standardization
-------------------------------------------------------------------------
This module provides a robust framework for extracting and standardizing 
Region of Interest (ROI) from pinwheel-style calibration targets. It implements 
automatic spatial orientation normalization, ensuring that edges from top, 
bottom, left, and right positions are uniformly transformed into a vertical 
orientation for consistent Spatial Frequency Response (SFR) calculation. 
The module facilitates the precise measurement of MTF50 and MTF@0.25 
across multiple axis to evaluate the overall sharpness of optical systems.

VIST-Compute-Core: 多向 SFR 分析与 ROI 标准化模块

本模块是 VIST 系统的核心计算单元之一，专门用于处理“风车式”标靶的图像数据。
它通过对 150x150 像素的原始 ROI 进行二次采样与几何变换，实现了边缘方向的标准化。

功能概述 (Functionality Overview):
-------------------------------
1. ROI 二次采样：从 150x150 的主图中精准提取四个方位（上下左右）的 32x32 子块。
2. 图像预处理：集成高斯滤波，用于抑制高频噪声并微量平滑信号，优化 SFR 计算稳定性。
3. 方向标准化：通过坐标旋转（如逆时针旋转 90 度），将水平边缘统一转换为垂直边缘，
   从而适配单一方向的 ISO 12233 计算逻辑。
4. 综合 MTF 评估：计算四个方位的 MTF@0.25 并求取均值，提供系统级的解析度反馈。
5. 结果可视化：集成 Matplotlib 绘图功能，直观展示每个方位的边缘分布与 MTF 响应曲线。

算法逻辑 (Logic Flow):
---------------------
[150x150 ROI] -> [Sub-sampling] -> [Gaussian Blur] -> [Rotation (if needed)] 
-> [ISO 12233 SFR Kernel] -> [MTF@0.25 Interpolation] -> [Global Averaging]

注意事项：
- 模块依赖外部算法库 `sfr_iso12233.py`，需确保其在同一路径下。
- 输入图像为线性响应的灰度图。
- 坐标偏移（Shift）和尺寸（Size）参数针对标准 150x150 风车标靶进行了硬编码优化。

作者: Zhang Lei
最后编辑时间: 2026-02-19
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from sfr_core.sfr_iso12233 import ISO12233_SFR
except ImportError:
    from sfr_iso12233 import ISO12233_SFR


def get_standardized_mtf_roi(img_150):
    """
    输入: 150x150 的风车 ROI (cv2.imread 结果)
    输出: 字典 {'top', 'bottom', 'left', 'right'}，
          每个 value 都是 32x32 的 uint8 矩阵，且边缘统一调整为【垂直方向】
    """
    if img_150 is None:
        return None
    
    # 固定的采样参数：基于 150x150 矩阵的几何中心进行定位
    mid = 75
    off = 25
    size = 32
    shift = 4 

    # 1. 定义四个方位的采样左上角坐标 (x1, y1)
    # top/bottom 负责提取垂直方向的空间频率信息（边缘为水平走向，需调整）
    # left/right 负责提取水平方向的空间频率信息
    roi_configs = {
        'top':    (mid + shift - size//2,  mid - off - size),
        'bottom': (mid - shift - size//2,  mid + off),
        'left':   (mid - off - size,       mid - shift - size//2),
        'right':  (mid + off,              mid + shift - size//2)
    }

    standardized_edges = {}

    for name, (x1, y1) in roi_configs.items():
        # 2. 截取原始 32x32 块
        x2, y2 = x1 + size, y1 + size
        # 边界安全保护：防止 ROI 超出 150 像素边界导致数组越界
        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(150, x2), min(150, y2)
        raw_roi = img_150[y1_c:y2_c, x1_c:x2_c].copy()
        # 针对 32x32 的 ROI 做微量去锐化
        # --- 关键预处理步骤：硬件锐化补偿 ---
        # 针对 Astra Pro 等相机设备，其硬件 ISP 锐化模块无法通过 UVC 协议完全关闭。
        # 这种“强行锐化”会在边缘产生 Overshoot（过冲），导致 MTF计算值虚高。
        # 此处使用 GaussianBlur (sigma=1.5) 进行反向平滑，旨在抑制锐化算子产生的高频伪影，
        # 从而还原更真实的镜头物理调制传递函数。
        raw_roi = cv2.GaussianBlur(raw_roi, (5, 5), 1.5)
        # 3. 方向标准化 (Direction Standardization)
        # 目标：让所有输出的矩阵中，黑白交界线都是【竖着的】（跨越 X 轴梯度）
        if name in ['top', 'bottom']:
            # top/bottom 在原始风车图中本身即跨越 X 轴呈现垂直方向边
            # 直接保持原样即可
            standardized_edges[name] = raw_roi
        else:
            # left/right 里的边缘是横着的（水平边缘），需要逆时针旋转 90 度变成竖着的
            # 统一算法入口，简化后续的信号投影逻辑
            standardized_edges[name] = cv2.rotate(raw_roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return standardized_edges
    
def get_MTF_at_025(img_150):
    """
    计算给定 ROI 图像在四个方向上的 MTF@0.25 均值。
    """
    try:
        mtf_inputs = get_standardized_mtf_roi(img_150)
        if mtf_inputs:
            # 分别计算四个方位的 MTF 指标
            # Top 方向计算
            sfr = ISO12233_SFR(mtf_inputs['top'])
            freqs, mtf, mtf50 = sfr.calculate_mtf()
            mtf_at_025_top = np.interp(0.25, freqs, mtf)

            # Bottom 方向计算
            sfr = ISO12233_SFR(mtf_inputs['bottom'])
            freqs, mtf, mtf50 = sfr.calculate_mtf()
            mtf_at_025_bottom = np.interp(0.25, freqs, mtf)

            # Left 方向计算
            sfr = ISO12233_SFR(mtf_inputs['left'])
            freqs, mtf, mtf50 = sfr.calculate_mtf()
            mtf_at_025_left = np.interp(0.25, freqs, mtf)

            # Right 方向计算
            sfr = ISO12233_SFR(mtf_inputs['right'])
            freqs, mtf, mtf50 = sfr.calculate_mtf()
            mtf_at_025_right = np.interp(0.25, freqs, mtf)
            
            # 返回四个方位的算术平均值，代表该点的综合清晰度
            return (mtf_at_025_top + mtf_at_025_bottom + mtf_at_025_left + mtf_at_025_right) / 4
        else:
            return 0
    except Exception as e:
        # 错误捕获：防止因个别区域边缘提取失败导致整个测试流程崩溃
        print(f"计算MTF时发生错误: {e}")
        return 0


# --- 使用示例 ---
if __name__ == '__main__':
    # 路径示例使用原始灰度读取
    img = cv2.imread(r'.\temp\roi4.png', 0)
    
    # 1. 快速获取 MTF 均值指标
    value = get_MTF_at_025(img)
    print(f"MTF 均值: {value:.4f}")
    
    # 2. 详细的四个方向分析与可视化
    mtf_inputs = get_standardized_mtf_roi(img)

    if mtf_inputs:
        for side, matrix in mtf_inputs.items():
            print(f"Edge {side} ready for MTF, shape: {matrix.shape}")

            # 调用底层 ISO12233 核心算法
            sfr = ISO12233_SFR(matrix)
            freqs, mtf, mtf50 = sfr.calculate_mtf()
            
            # 提取 0.25 cy/pixel 处的值（关键中频响应指标）
            mtf_at_025 = np.interp(0.25, freqs, mtf)
            
            print("\n" + "="*40)
            print(f"仿真结果报告 ({side})")
            print(f"MTF50 频率值: {mtf50:.4f} cy/px")
            print(f"关键指标 MTF@0.25: {mtf_at_025:.4f}")
            print("="*40)

            # 结果可视化对比图绘制
            plt.figure(figsize=(12, 5))
            
            # 左侧：显示标准化后的边缘子块
            plt.subplot(1, 2, 1)
            plt.imshow(matrix, cmap='gray')
            plt.title(f"Standardized Edge ROI: {side}")
            
            # 右侧：绘制 MTF 曲线
            plt.subplot(1, 2, 2)
            plt.plot(freqs, mtf, 'b-', linewidth=2, label='MTF Curve')
            plt.axvline(0.25, color='r', linestyle='--', label='Target Freq 0.25')
            plt.scatter([0.25], [mtf_at_025], color='red', s=100)
            plt.text(0.26, mtf_at_025 + 0.05, f'Value: {mtf_at_025:.3f}', color='red', fontweight='bold')
            
            plt.ylim(0, 1.1)
            plt.xlim(0, 0.5)
            plt.xlabel("Frequency (cy/px)")
            plt.ylabel("Modulation")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # 导出标准化后的边缘图像以便离线核验
            cv2.imwrite(fr'.\temp\ready_to_mtf_{side}.png', matrix)