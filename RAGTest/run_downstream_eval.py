import os
import argparse
import random
import numpy as np
import torch
from llama_index.core import Settings, PromptTemplate
from llms.llm import get_llm
from index import get_index
from embs.embedding import get_embedding
from data.qa_loader import get_qa_dataset
from config import Config
from retriever import get_retriver, response_synthesizer
from eval.evaluate_rag import NLGEvaluate
from llama_index.core.query_engine import RetrieverQueryEngine

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def hit(retrieval_ids, golden_context_ids, k=1):
    for golden_id in golden_context_ids:
        if golden_id in retrieval_ids[:k]:
            return 1
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model.")
    args = parser.parse_args()

    seed_everything(42)
    cfg = Config()

    print(f"Loading model from: {args.model_path}")
    embeddings = get_embedding(args.model_path)

    qa_dataset = get_qa_dataset(cfg.dataset)
    llm = get_llm(cfg.llm)

    Settings.chunk_size = cfg.chunk_size
    Settings.llm = llm
    Settings.embed_model = embeddings

    persist_dir = f"./index_cache-{os.path.basename(args.model_path)}"
    index, hierarchical_storage_context = get_index(qa_dataset, persist_dir, split_type=cfg.split_type, chunk_size=cfg.chunk_size)

    query_engine = RetrieverQueryEngine(
        retriever=get_retriver(cfg.retriever, index, hierarchical_storage_context=hierarchical_storage_context),
        response_synthesizer=response_synthesizer(0),
    )

    text_qa_template_str = (
        "Based solely on the context below, and without using ANY prior knowledge, "
        "answer the following question as concisely as possible: {query_str}\\n"
        "Context: {context_str}\\n"
    )
    text_qa_template = PromptTemplate(text_qa_template_str)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": text_qa_template})

    total_hit_1 = 0
    total_hit_3 = 0
    total_hit_5 = 0
    num_questions = 0

    for question, expected_answer, golden_context, golden_context_ids in zip(
            qa_dataset['question'],
            qa_dataset['answers'],
            qa_dataset['golden_sentences'],
            qa_dataset['golden_ids']
    ):
        num_questions += 1
        print(f"\\n--- Question {num_questions} ---")
        print(f"Question: {question}")

        response_nodes = query_engine.retrieve(question)

        retrieval_ids = [node.metadata['id'] for node in sorted(response_nodes, key=lambda x: x.score, reverse=True)]

        total_hit_1 += hit(retrieval_ids, golden_context_ids, k=1)
        total_hit_3 += hit(retrieval_ids, golden_context_ids, k=3)
        total_hit_5 += hit(retrieval_ids, golden_context_ids, k=5)

        print(f"Retrieved IDs: {retrieval_ids}")
        print(f"Golden IDs: {golden_context_ids}")
        print(f"Hit@1: {total_hit_1 / num_questions:.4f}, Hit@3: {total_hit_3 / num_questions:.4f}, Hit@5: {total_hit_5 / num_questions:.4f}")

    print("\\n--- Final Results ---")
    print(f"Hit@1: {total_hit_1 / num_questions:.4f}")
    print(f"Hit@3: {total_hit_3 / num_questions:.4f}")
    print(f"Hit@5: {total_hit_5 / num_questions:.4f}")

if __name__ == '__main__':
    main()
