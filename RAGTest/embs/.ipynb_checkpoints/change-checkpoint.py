import torch
import os
from transformers import AutoModel,AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
import os

def fix_model_state_dict(model_path):
 # 1. 查找.pt文件
    pt_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    if not pt_files:
        print(f"No .pt files found in {model_path}")
        return

    print(pt_files[0])
    
    checkpoint_path = os.path.join(model_path, pt_files[0])  # 使用找到的第一个.pt文件
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    print(state_dict.keys())
    
    # 2. 创建新的状态字典，去除 'model.' 前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # 删除 'model.' 前缀
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    print(123)
    print(new_state_dict.keys())
    
    # 3. 保存修改后的状态字典
    backup_path = os.path.join(model_path, 'pytorch_model.bin')

    torch.save(new_state_dict, backup_path)
    print(f"Model weights have been fixed and saved. Original file backed up as {backup_path}")

    print("embedding_path: ", model_path)
    embeddings = HuggingFaceEmbedding(
        model_name=model_path,
    )

    print("success")

# 使用示例
model_path = "/root/autodl-tmp/model/zh-model-num4"
fix_model_state_dict(model_path)
