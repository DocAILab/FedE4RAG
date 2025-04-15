from llama_index.llms.openai import OpenAI

from llms.huggingface_model import get_huggingfacellm
from llms import chatglm4
from embs import chatglmemb
from config import Config
from llama_index.llms.ollama import Ollama
import os

llm_dict = {
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "chatglm": "THUDM/chatglm3-6b",
    "qwen": "Qwen/Qwen1.5-7B-Chat",
    "qwen14_int8": "Qwen/Qwen1.5-14B-Chat-GPTQ-Int8",
    "qwen7_int8": "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
    "qwen1.8": "Qwen/Qwen1.5-1.8B-Chat",
    "baichuan": "baichuan-inc/Baichuan2-7B-Chat",
    "falcon": "tiiuae/falcon-7b-instruct",
    "mpt": "mosaicml/mpt-7b-chat",
    "yi": "01-ai/Yi-6B-Chat",
}


def get_openai(api_base,api_key,api_name):
    return OpenAI(api_key=api_key,api_base=api_base, temperature=0,model=api_name)

ollama_url = "http://223.70.119.218:20012"
os.environ['OLLAMA_HOST'] = ollama_url
def get_llm(name):
    if name in llm_dict.keys():
        return get_huggingfacellm(llm_dict[name])
    elif name == 'chatgpt-3.5':
        return get_openai(Config().api_base,Config().api_key,Config().api_name)
    elif name == 'gpt-4o-mini':
        return get_openai(Config().api_base,Config().api_key,Config().api_name)
    elif name == 'Llama3.1:8B':
        return Ollama(model="Llama3.1:8B", request_timeout=3600, base_url=ollama_url)
    elif name == 'deepseek-r1:7b':
        return Ollama(model="deepseek-r1:7b", request_timeout=3600, base_url=ollama_url)
    elif name == 'openthinker:latest':
        return Ollama(model="openthinker:latest", request_timeout=3600, base_url=ollama_url)
    elif name == 'mathstral:latest':
        return Ollama(model="mathstral:latest", request_timeout=3600, base_url=ollama_url)
    else:
        raise ValueError(f"no model name: {name}.")


