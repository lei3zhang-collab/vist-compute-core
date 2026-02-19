"""
VIST Compute Core - High-Speed UDP Data Bridge & MTF Analysis Engine
-------------------------------------------------------------------------
This module implements a high-performance UDP receiver and processing bridge 
designed for synchronized image data transfer between ROS2 (running in a 
Linux VM) and a Windows-based PC. It handles fragmented luminance data 
reconstruction for 9 distinct Regions of Interest (ROIs), executes sub-pixel 
Spatial Frequency Response (SFR) analysis, and facilitates the low-latency 
return of MTF50/MTF@0.25 results back to the source node for real-time 
system calibration.

VIST-Compute-Core: 高速 UDP 数据桥接与 MTF 分析引擎

本模块是 VIST 自动化测试系统的跨系统通讯与计算核心。它主要解决了在 Ubuntu 
虚拟机（运行 ROS2）与 Windows 主机（执行高负载运算）之间的无损图像数据传输。

功能概述 (Functionality Overview):
-------------------------------
1. 高可靠性 UDP 接收：通过自定义包头协议（0xEE）实现分片数据的无损重组与偏移校准。
2. 亮度矩阵还原：将 202,500 字节的二进制流还原为 9 个标准的 150x150 亮度矩阵。
3. 批量 MTF 计算：调用底层 SFR 内核对 9 个方位（风车标靶）的 ROI 进行解析度量化。
4. 结果闭环回传：利用 Socket 二进制流将计算出的 MTF@0.25 指标回传至虚拟机，实现闭环控制。
5. 状态自修复：具备缓冲区重置与状态清理机制，支持连续、长时间的产线自动化测试。

技术路径：
[VM ROS2 Node] --(UDP Fragment)--> [This PC Bridge] --(Compute)--> [MTF Results] --(UDP)--> [VM]

作者: Zhang Lei
最后编辑时间: 2026-02-19
"""

import socket
import numpy as np
import cv2
from sfr_core import mtf_engine

# --- 回传配置 (Return Configuration) ---
# 用于将计算结果发送到虚拟机的专用套接字
VM_IP = "192.168.106.129"  # 虚拟机目标 IP 地址
RETURN_PORT = 5006         # 虚拟机接收 MTF 结果的端口
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- 接收配置 (Receive Configuration) ---
# 与 ROS2 端发送节点严格对齐的参数
UDP_IP = "0.0.0.0"       # 监听本地所有可用网卡
UDP_PORT = 5005          # 接收原始图像数据的端口
EXPECTED_TOTAL = 202500  # 预期总长度: 9 个方块 * 150 宽 * 150 高
CROP_SIZE = 150          # 单个 ROI 的边长

# 初始化接收套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
# Win11 ，适配设置超时，使阻塞操作能被 KeyboardInterrupt 打断
sock.settimeout(1.0) 


# 状态变量初始化
buffer = bytearray(EXPECTED_TOTAL)
received_offsets = set()

print(f"VIST PC 端就绪，等待无损亮度数据 (端口:{UDP_PORT})...")
def main_loop():
    global buffer, received_offsets

    while True:
        try:
            # 尝试接收数据
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            # 超时后循环继续，这给了 Python 解释器处理 Ctrl+C 信号的机会
            continue

        # 1. 数据包解析 (Packet Header Parsing)
        # 包头结构: [1字节标志 0xEE] + [4字节偏移 Offset] + [4字节总长 Total]
        if data[0] == 0xEE:
            offset = int.from_bytes(data[1:5], 'big')
            total_len = int.from_bytes(data[5:9], 'big')
            payload = data[9:]
            
            # 填充静态缓冲区
            buffer[offset : offset + len(payload)] = payload
            received_offsets.add(offset)
            
            # 2. 完整性检查与处理触发
            # 基于包数量与头部有效性进行简单判断
            if len(received_offsets) * 50000 >= EXPECTED_TOTAL or sum(len(p) for p in [payload]) == EXPECTED_TOTAL: 
                # 粗略检查缓冲区头部是否已被填充
                if all(buffer[i] != 0 for i in range(0, 10)): 
                    print(">>> 9 图数据收齐，开始还原与计算...")
                    
                    # 3. 数据还原 (Data Reconstruction)
                    raw_data = np.frombuffer(buffer, dtype=np.uint8)
                    all_rois = raw_data.reshape(9, CROP_SIZE, CROP_SIZE)
                    
                    # --- [DEBUG CODE BLOCK: 图像保存与可视化] ---
                    # 提示：若需核验传输质量，可取消以下代码注释
                    """
                    # 拼成 3x3 大图查看效果
                    rows = [np.hstack(all_rois[i:i+3]) for i in range(0, 9, 3)]
                    combined = np.vstack(rows)
                    cv2.imshow("PC Received MTF ROIs (Luminance)", combined)
                    cv2.imwrite("received_mtf_rois.png", combined)
                    cv2.waitKey(1)
                    
                    # 保存单个方块用于离线算法分析
                    for idx in range(9):
                        cv2.imwrite(fr".\temp\roi{idx}.png", all_rois[idx])
                    """
                    # --------------------------------------------
                    
                    # 4. 批量 MTF 计算 (MTF Computation)
                    # 调用底层 mtf_engine 进行各方位 MTF@0.25 提取
                    mtf_values = [mtf_engine.get_MTF_at_025(roi) for roi in all_rois]
                    print(f"MTF 计算完成: {mtf_values}")

                    # 5. 数据封装与回传 (Feedback Loop)
                    # 将 9 个浮点数转为二进制流 (float32，总计 36 字节)
                    mtf_array = np.array(mtf_values, dtype=np.float32)
                    mtf_bytes = mtf_array.tobytes()
                    
                    # 将结果精确回传至虚拟机
                    send_sock.sendto(mtf_bytes, (VM_IP, RETURN_PORT))
                    print(f">>> MTF 结果已回传至 {VM_IP}:{RETURN_PORT}")

                    # 6. 状态清理 (Cleanup for Next Cycle)
                    # 确保下一帧数据不受当前缓冲区残留影响
                    received_offsets.clear()
                    buffer = bytearray(EXPECTED_TOTAL) 

if __name__ == '__main__':
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[VIST] 检测到用户中断 (Ctrl+C)。")
    finally:
        # 资源释放 (Resource Management)
        print("[VIST] 正在清理 Socket 资源并释放端口...")
        sock.close()
        send_sock.close()
        print("[VIST] 退出成功。")
        # sys.exit(0)