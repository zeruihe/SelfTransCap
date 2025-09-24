#负责根据配置文件中指定的路径和格式读取数据
# src/data_processing/loader.py

import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable

class TransCapDataset(Dataset):
    """
    一个PyTorch Dataset类，用于加载由TransCap方法生成的图像-文本对数据。
    
    该类实现了用户请求的核心功能：在每次获取样本时，从为该图像生成的
    多个描述中随机选择一个，这在训练过程中起到了一种数据增强的作用。
    """
    def __init__(self, 
                 captions_csv: str, 
                 image_dir: str, 
                 transforms: Optional[Callable] = None):
        """
        初始化数据集。

        Args:
            captions_csv (str): 包含图像ID和多个描述列的CSV文件路径。
            image_dir (str): 存储PNG格式图像的目录路径。
            transforms (Optional[Callable], optional): 应用于图像的torchvision变换。
        """
        self.captions_df = pd.read_csv(captions_csv)
        self.image_dir = image_dir
        self.transforms = transforms
        
        # 动态获取所有 'describe_' 开头的列
        self.caption_columns = [col for col in self.captions_df.columns if col.startswith('describe_')]
        if not self.caption_columns:
            raise ValueError("No 'describe_' columns found in the captions CSV file.")

    def __len__(self) -> int:
        """返回数据集中的样本总数。"""
        return len(self.captions_df)

    def __getitem__(self, idx: int) -> (torch.Tensor, str):
        """
        获取一个样本（图像和随机选择的描述）。

        Args:
            idx (int): 样本的索引。

        Returns:
            (torch.Tensor, str): 转换后的图像张量和一条随机选择的文本描述。
        """
        row = self.captions_df.iloc[idx]
        patient_id = row['patientId']
        
        # 随机选择一个描述
        selected_caption_col = random.choice(self.caption_columns)
        caption = row[selected_caption_col]
        
        # 加载图像
        img_path = os.path.join(self.image_dir, f"{patient_id}.png")
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path}. Skipping.")
            # 在实际应用中，可能需要更优雅地处理缺失文件
            # 这里我们返回一个占位符或引发异常
            # 为简单起见，我们假设所有文件都存在
            # 如果要健壮，可以重新尝试加载下一个有效的样本
            return self.__getitem__((idx + 1) % len(self))

        # 应用图像变换
        if self.transforms:
            image = self.transforms(image)
            
        return image, str(caption)