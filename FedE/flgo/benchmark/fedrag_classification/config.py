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
    # Using a smaller, VRAM-friendly model to fit within Colab's memory constraints.
    # The original model was too large for the T4 GPU environment.
    model = BertModel.from_pretrained('BAAI/bge-small-en-v1.5')
    return model