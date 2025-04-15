# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModel,AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
import os

def get_embedding(state_dict_path):    
    print("embedding_path: ", state_dict_path)

    embeddings = HuggingFaceEmbedding(
        model_name=state_dict_path,
        # embed_batch_size=128,
    )
    return embeddings

'''
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def get_embedding(name):
    encode_kwargs = {"batch_size": 128, 'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(
        model_name=name,
        encode_kwargs=encode_kwargs,
        # embed_batch_size=128,
    )
    return embeddings
'''