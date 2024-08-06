import os

import numpy as np
import psutil
import torch


def list_to_colormap(x_list):  # IP:21025
    y = np.zeros((x_list.shape[0], 3))  # IP:(21025,3)
    label_color_IP = {
        0: np.array([0, 0, 0]),  # 黑色
        1: np.array([255, 192, 203]),  # 浅粉红
        2: np.array([152, 251, 152]),  # 淡绿
        3: np.array([135, 206, 250]),  # 天蓝
        4: np.array([255, 255, 224]),  # 米黄
        5: np.array([224, 255, 255]),  # 明青
        15: np.array([238, 130, 238]),  # 紫罗兰
        6: np.array([211, 211, 211]),  # 银白
        16: np.array([191, 191, 191]),  # 轻灰
        7: np.array([255, 228, 181]),  # 浅橙
        8: np.array([255, 182, 193]),  # 桃粉
        9: np.array([205, 133, 63]),  # 栗色
        10: np.array([255, 215, 0]),  # 金黄
        11: np.array([199, 97, 20]),  # 栗棕
        12: np.array([152, 251, 152]),  # 淡绿
        13: np.array([255, 255, 255]),  # 白色
        14: np.array([240, 240, 240]),  # 浅灰色
    }
    label_color_IP = {
        0: np.array([0, 0, 0]),
        14: np.array([147, 67, 46]),
        1: np.array([0, 0, 255]),
        2: np.array([255, 100, 0]),
        3: np.array([0, 255, 123]),
        4: np.array([164, 75, 155]),
        5: np.array([101, 174, 255]),
        15: np.array([118, 254, 172]),
        6: np.array([60, 91, 112]),
        16: np.array([255, 255, 0]),
        7: np.array([255, 255, 125]),
        8: np.array([255, 0, 255]),
        9: np.array([100, 0, 255]),
        10: np.array([0, 172, 254]),
        11: np.array([0, 255, 0]),
        12: np.array([171, 175, 80]),
        13: np.array([101, 193, 60])
    }
    label_color = {
        0: np.array([0, 0, 0]),
        1: np.array([147, 67, 46]),
        2: np.array([0, 0, 255]),
        3: np.array([255, 100, 0]),
        4: np.array([0, 255, 123]),
        5: np.array([164, 75, 155]),
        6: np.array([101, 174, 255]),
        7: np.array([118, 254, 172]),
        8: np.array([60, 91, 112]),
        9: np.array([255, 255, 0]),
        10: np.array([255, 255, 125]),
        11: np.array([255, 0, 255]),
        12: np.array([100, 0, 255]),
        13: np.array([0, 172, 254]),
        14: np.array([0, 255, 0]),
        15: np.array([171, 175, 80]),
        16: np.array([101, 193, 60])
    }
    label_color = label_color
    for index, item in enumerate(x_list):
        if item != 0:
            y[index] = label_color[item] / 255.0

    return y


def gpu():
    # 获取GPU设备数量
    num_gpu = torch.cuda.device_count()
    # 获取当前使用的GPU索引
    current_gpu_index = torch.cuda.current_device()
    # 获取当前GPU的名称
    current_gpu_name = torch.cuda.get_device_name(current_gpu_index)
    # 获取GPU显存的总量和已使用量
    total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory / (1024 ** 3)  # 显存总量(GB)
    used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)  # 已使用显存(GB)
    free_memory = total_memory - used_memory  # 剩余显存(GB)
    print(f"CUDA可用，共有 {num_gpu} 个GPU设备可用。")
    print(f"当前使用的GPU设备索引：{current_gpu_index}")
    print(f"当前使用的GPU设备名称：{current_gpu_name}")
    print(f"GPU显存总量：{total_memory:.2f} GB")
    print(f"已使用的GPU显存：{used_memory:.2f} GB")
    print(f"剩余GPU显存：{free_memory:.2f} GB")


def cpu():
    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
