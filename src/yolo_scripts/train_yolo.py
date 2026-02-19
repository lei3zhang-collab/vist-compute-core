"""
VIST Compute Core - YOLO-based Target Detection and Training Pipeline
-------------------------------------------------------------------------
This module implements the deep learning training workflow for the VIST 
automatic calibration system. It leverages the Ultralytics YOLO framework 
to train a lightweight object detection model (YOLO26n) specifically 
optimized for identifying MTF pinwheel targets and slanted-edge ROIs. 
The pipeline includes data augmentation and rectangular training strategies 
to ensure high recall and precision even with limited specialized datasets, 
laying the foundation for subsequent sub-pixel SFR analysis.

VIST-Compute-Core: 基于 YOLO 的标靶检测训练流水线

本模块为 VIST 系统的前端感知层训练代码。在执行 ISO 12233 SFR 算法之前，系统需要
通过深度学习模型从复杂的工业环境图像中自动定位标靶区域（ROI）。

功能概述 (Functionality Overview):
-------------------------------
1. 模型加载：选用 Nano 系列轻量化模型，确保在测试现场的嵌入式或
   PC 端设备上实现毫秒级的推理响应。
2. 训练配置：针对 MTF 标靶特征，采用了 1280 像素的高分辨率输入，
   以保留边缘的空间细节，防止小目标丢失。
3. 鲁棒性优化：开启数据增强（Augment）和矩形训练（Rect），
   有效应对生产线上可能出现的图像旋转、光照变化及长宽比失真。
4. 闭环验证：训练完成后生成的权重文件将直接驱动后端的自动裁切逻辑，从而实现
   “检测 -> 裁切 -> MTF 计算” 的全自动化测试链路。

技术要点 (Technical Highlights):
------------------------------
- 硬件适配：默认配置为 CPU 训练，兼顾兼容性。在具备 CUDA 条件的环境下可自动切换。
- 策略平衡：通过限制 Epochs 数和 Batch Size，在有限的样本量下平衡过拟合风险与收敛速度。

作者: Zhang Lei
最后编辑时间: 2026-02-19
"""

from ultralytics import YOLO

# 1. 加载预训练模型 (使用 Nano 版本，检测速度最快)

model = YOLO('yolo26n.pt') 

# 2. 开始训练

model.train(
    data=r'.\data\mtf_dataset\mtf_data.yaml',
    epochs=100,         # 样本较少时，100 轮通常能达到较好的收敛效果
    imgsz=1280,          # 训练缩放尺寸：调大尺寸有助于捕捉边缘的微小特征
    rect=True,           # 开启矩形训练：减少无效填充（Padding），提升训练效率
    batch=8,            # 显存/内存占用优化设置
    save=True,          # 训练结束后自动保存最佳权重 (best.pt)
    name='mtf_720p_v1',      # 本次训练任务的实验名称，结果将保存至 runs/detect/ 目录下
    augment=True,        # 开启数据增强：通过随机变换弥补实拍图片不足的问题，提高模型泛化力
    device='cpu'        # 强制指定 CPU 训练，确保在无独立显卡的测试机上平稳运行
)
