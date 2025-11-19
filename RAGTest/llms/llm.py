from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from llama_index.llms.huggingface import HuggingFaceLLM

def get_llm(model_name="meta-llama/Llama-2-7b-chat-hf"):
    """
    Loads a 4-bit quantized version of a Llama 8B model from Hugging Face.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.1, "do_sample": False},
        query_wrapper_prompt=None, # No query wrapper prompt
    )

    return llm
