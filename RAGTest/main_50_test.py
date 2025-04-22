import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from llama_index.core import Settings, PromptTemplate
from llms.llm import get_llm
from index import get_index
from eval.evaluate_rag import EvaluationResult, NLGEvaluate
from embs.embedding import get_embedding
from data.qa_loader import get_qa_dataset
from config import Config
from retriever import *
from eval.evaluate_TRT import EvaluationResult_TRT
from eval.evaluate_TGT import evaluating_TGT
from eval.evaluate_TRT import evaluating_TRT
from eval.EvalModelAgent import EvalModelAgent
from process.postprocess_rerank import get_postprocessor
from process.query_transform import transform_and_query
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(42)

name = "Your LLM api"
auth_token = "Your api key"



cfg = Config()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = None)
args = parser.parse_args()
if args.model is None:
    print("no model path, use default path——bge")
    embeddings = get_embedding("/root/autodl-tmp/model/model-en")
    last_dir = "BGE-en-50-test"
else:
    embeddings = get_embedding(args.model)
    last_dir = os.path.basename(args.model)
qa_dataset = get_qa_dataset(cfg.dataset)
print("dataset")
print(last_dir)

llm = get_llm(cfg.llm)
print("llm")

# Create and dl embeddings instance


Settings.chunk_size = cfg.chunk_size
Settings.llm = llm
Settings.embed_model = embeddings
# pip install llama-index-embeddings-langchain

cfg.persist_dir = cfg.persist_dir + '-' + cfg.dataset + '-' + last_dir + '-' + cfg.split_type + '-' + str(cfg.chunk_size)

index, hierarchical_storage_context = get_index(qa_dataset, cfg.persist_dir, split_type=cfg.split_type, chunk_size=cfg.chunk_size)
print("index")

query_engine = RetrieverQueryEngine(

    retriever=get_retriver(cfg.retriever, index, hierarchical_storage_context=hierarchical_storage_context), # todo: cfg.retriever
    response_synthesizer=response_synthesizer(0),
    node_postprocessors=[get_postprocessor(cfg)]
)

text_qa_template_str = (
    "Below is the context information.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Based solely on the above context, not prior knowledge, please answer the following question: {query_str}\n"
    "Instructions: Keep your answer extremely brief. Focus only on the most essential information from the context."
)
text_qa_template = PromptTemplate(text_qa_template_str)

# Setup index query engine using LLM
# query_engine = index.as_query_engine(response_mode="compact")

query_engine.update_prompts({"response_synthesizer:text_qa_template": text_qa_template})
query_engine = query_expansion([query_engine], query_number=4, similarity_top_k=10)
query_engine = RetrieverQueryEngine.from_args(query_engine)


# question = "Which team does the player named 2015 Diamond Head Classic’s MVP play for?"
# answer = "Sacramento Kings"
# golden_source = "The 2015 Diamond Head Classic was a college:    basketball tournament ... Buddy Hield was named the tournament’s MVP. Chavano Rainier ”Buddy” Hield is a Bahamian professional basketball player for the Sacramento Kings of the NBA..."
true_num = 0
all_num = 0
evaluateResults_TRT = EvaluationResult_TRT()

evalAgent = EvalModelAgent(cfg)
if cfg.experiment_1:
    if len(qa_dataset) < cfg.test_init_total_number_documents:
        warnings.filterwarnings('default')
        warnings.warn("使用的数据集长度大于数据集本身的最大长度，请修改。 本轮代码无法运行", UserWarning)
else:
    cfg.test_init_total_number_documents = cfg.n
    
"""
hzt todo
"""
evaluateResults_TGT = EvaluationResult(metrics=[
    "NLG_chrf", "NLG_bleu", "NLG_meteor", "NLG_wer", "NLG_cer", "NLG_chrf_pp",
                               "NLG_mauve", "NLG_perplexity",
                               "NLG_rouge_rouge1", "NLG_rouge_rouge2", "NLG_rouge_rougeL", "NLG_rouge_rougeLsum"
    # "Llama_retrieval_Faithfulness", "Llama_retrieval_Relevancy", "Llama_response_correctness",
    #                           "Llama_response_semanticSimilarity", "Llama_response_answerRelevancy","Llama_retrieval_RelevancyG",
    #                           "Llama_retrieval_FaithfulnessG",
    #                           "DeepEval_retrieval_contextualPrecision","DeepEval_retrieval_contextualRecall",
    #                           "DeepEval_retrieval_contextualRelevancy","DeepEval_retrieval_faithfulness",
    #                           "DeepEval_response_answerRelevancy","DeepEval_response_hallucination",
    #                           "DeepEval_response_bias","DeepEval_response_toxicity",
    #                           "UpTrain_Response_Completeness","UpTrain_Response_Conciseness","UpTrain_Response_Relevance",
    #                           "UpTrain_Response_Valid","UpTrain_Response_Consistency","UpTrain_Response_Response_Matching",
    #                           "UpTrain_Retrieval_Context_Relevance","UpTrain_Retrieval_Context_Utilization",
    #                           "UpTrain_Retrieval_Factual_Accuracy","UpTrain_Retrieval_Context_Conciseness",
    #                           "UpTrain_Retrieval_Code_Hallucination",

])
    
# 初始化全局字典和计数器
global_evaluation_scores = {}
evaluation_count = 0

def add_scores(new_scores):
    global global_evaluation_scores, evaluation_count
    
    # 更新计数器
    evaluation_count += 1
    
    # 累加新的评估结果到全局字典中
    for metric, score in new_scores.items():
        if metric not in global_evaluation_scores:
            global_evaluation_scores[metric] = 0
        global_evaluation_scores[metric] += score

def compute_and_format_averages():
    global global_evaluation_scores, evaluation_count
    
    # 计算平均值并格式化输出
    formatted_output = "NLG Evaluation Metrics Averages:\n"
    for metric, total_score in global_evaluation_scores.items():
        average_score = total_score / evaluation_count
        formatted_output += f"{metric}: {average_score:.4f}\n"
    
    return formatted_output
    
    
for question, expected_answer, golden_context, golden_context_ids in zip(
        qa_dataset['question'],
        qa_dataset['answers'],
        qa_dataset['golden_sentences'],
        qa_dataset['golden_ids']
        ):

        print(question)
        print(expected_answer)
        print(golden_context)

        # response = transform_and_query(question, cfg, query_engine)
        response = query_engine.retrieve(question)
        print(response)
        # 返回node节点
        retrieval_ids = []
        retrieval_context = []
        for source_node in response: #.source_nodes:
            retrieval_ids.append(source_node.metadata['id'])
            retrieval_context.append(source_node.get_content())
        # actual_response = response.response
        actual_response = ""
        print(actual_response)
        print(retrieval_ids, "--- hzt ---", golden_context_ids)
        print(response)
        # print(response.source_nodes)
        eval_result = evaluating_TRT(retrieval_ids, golden_context_ids)
        evaluateResults_TRT.add(eval_result)

        # score = NLGEvaluate(actual_response, expected_answer)
        # add_scores(score)
        print(compute_and_format_averages())
        
        all_num = all_num + 1
        
        print("TRT",'-'*100)
        evaluateResults_TRT.print_results()
        print("TRT",'-'*100)
        
        print("总数：" + str(all_num))

evaluateResults_TRT.print_results_to_path("./50_test_TRT.txt", cfg, last_dir)
# python main.py --evaluateApiName="gpt-3.5-turbo" --evaluateApiKey="sk-FDMbb0bnifWopAgv4501226dEa4c45E5Ac81841123Eb80B0"
if __name__ == '__main__':
    print('Success')

