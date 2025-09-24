# train_example.py

import random
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.data_processing.loader import TransCapDataset

# --- 设置 ---
# 为了实验可复现性，固定随机种子
torch.manual_seed(42)
random.seed(42)

# --- 参数 ---
CAPTIONS_CSV_PATH = 'outputs/data/image_captions.csv'
IMAGE_DIR_PATH = 'data/raw/stage_2_train_images_png/'
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 10

# 1. 定义图像预处理/增强变换
#    这些变换应与您下游模型的预训练设置相匹配
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 2. 实例化数据集
print("Initializing TransCapDataset...")
train_dataset = TransCapDataset(
    captions_csv=CAPTIONS_CSV_PATH,
    image_dir=IMAGE_DIR_PATH,
    transforms=image_transforms
)
print(f"Dataset initialized with {len(train_dataset)} samples.")

# 3. 创建DataLoader
#    DataLoader负责批量化、打乱数据和多进程加载
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True # 如果使用GPU，可以加速数据传输
)

# 4. 在训练循环中迭代数据
print("Starting training loop example...")
for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    
    for i, (images, captions) in enumerate(train_loader):
        # 'images' 是一个形状为 (batch_size, channels, height, width) 的张量
        # 'captions' 是一个长度为 BATCH_SIZE 的元组，包含随机选择的文本描述
        
        # 在这里执行您的模型训练逻辑
        # 例如:
        # model.train()
        # optimizer.zero_grad()
        #
        # outputs = model(images, captions)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
        
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(train_loader)}")
            print(f"    Image batch shape: {images.shape}")
            print(f"    First caption in batch: '{captions[0]}'")

print("\nTraining loop example finished.")