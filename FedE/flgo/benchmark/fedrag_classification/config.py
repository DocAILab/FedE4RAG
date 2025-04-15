"""
train_data (torch.utils.data.Dataset),
test_data (torch.utils.data.Dataset),
and the model (torch.nn.Module) should be implemented here.

"""
import torch.nn
from transformers import BertModel

train_data = None
val_data = None
test_data = None
vocab = None
tokenizer = None

def get_model(*args, **kwargs) -> torch.nn.Module:
    # TODO 加载embedding模型 在largemodel
    model = BertModel.from_pretrained('/home/fedrag/bge-en')
    return model