"""
VIST Compute Core - YOLO Model Export and Deployment Optimization
-------------------------------------------------------------------------
This module handles the transition from trained PyTorch models to production-ready 
ONNX formats. It optimizes the inference graph for target environments such as 
AMD CPU-based industrial PCs and Linux virtual machines using ROS2. The export 
pipeline emphasizes spatial consistency for 720p resolution, operator 
simplification for reduced computational overhead, and compatibility with 
ONNX Runtime (Opset 12), ensuring stable real-time detection of MTF 
calibration targets.

VIST-Compute-Core: YOLO 模型导出与部署优化模块

本模块负责将训练完成的 PyTorch 模型 (.pt) 转换为工业界通用的 ONNX 格式。
这是实现自动化测试平台跨平台部署（如从 Windows 开发环境到 Linux 虚拟机）的核心步骤。

功能概述 (Functionality Overview):
-------------------------------
1. 算子优化：通过简化（Simplify）技术消除冗余算子，降低 AMD 等 CPU 平台的计算负载。
2. 尺寸标准化：针对 720P 工业相机视场进行显式尺寸锁定，防止在推理阶段产生比例畸变。
3. 兼容性对齐：采用 Opset 12 标准，确保在较低版本的 ONNX Runtime 或 Ubuntu 环境下顺利加载。
4. 部署准备：优化后的模型将用于 VIST 系统的自动裁切环节，为后端的 SFR 计算提供稳定的 ROI 输入。

技术要点 (Technical Highlights):
------------------------------
- 精度控制：在 CPU 链路上保持 FP32 精度，确保对于边缘细节敏感的标靶检测具有极高的鲁棒性。
- 架构固定：关闭动态尺寸以换取推理耗时的确定性，这在自动化流水线的时序控制中至关重要。

作者: Zhang Lei
最后编辑时间: 2026-02-19
"""

from ultralytics import YOLO

# 1. 加载训练完成的权重文件
# 将最终选定的权重文件拷贝至专用的 /models 文件夹

model = YOLO(r'.\models\best.pt')

# 2. 执行导出任务 (Export to ONNX)
# 针对 AMD CPU 推理及 Ubuntu 虚拟机环境进行了深度配置：
path = model.export(
    format='onnx',      # 导出目标格式：工业界互操作性最强的 Open Neural Network Exchange
    imgsz=[720, 1280],  # 尺寸锁定：明确 [高, 宽]，匹配 720p 相机原生分辨率，消除推理时的自动 Padding 干扰
    simplify=True,      # 算子融合：合并卷积与 BN 层，减少推理计算图的层数，对 CPU 推理非常友好
    opset=12,           # 算子集版本：选择稳定性与兼容性最佳平衡点的版本，适配多种后端 Runtime
    dynamic=False,      # 静态结构：工业相机位姿通常固定，关闭动态输入可获得更稳定的内存分配与推理帧率
    half=False          # 精度模式：CPU 环境下 FP16 往往由于指令集限制反而更慢，故坚持使用 FP32 保证数值精度
)

# 3. 状态反馈
# 提示：导出后的 .onnx 文件通常位于与 .pt 同级的文件夹内。
print(f"✅ 导出成功！请将该文件拷贝到虚拟机进行部署验证: {path}")
