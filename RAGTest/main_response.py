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
parser.add_argument('--model', type=str, default=None)
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
# test_50_1
# 100_6066
# 100_all
cfg.persist_dir = cfg.persist_dir + '-' + cfg.dataset + '-' + last_dir + '100_50_6066' + '-' + cfg.split_type + '-' + str(
    cfg.chunk_size)

index, hierarchical_storage_context = get_index(qa_dataset, cfg.persist_dir, split_type=cfg.split_type,
                                                chunk_size=cfg.chunk_size)
print("index")

query_engine = RetrieverQueryEngine(
    retriever=get_retriver(cfg.retriever, index, hierarchical_storage_context=hierarchical_storage_context),
    # todo: cfg.retriever
    response_synthesizer=response_synthesizer(0),
    node_postprocessors=[get_postprocessor(cfg)]
)


def hit(retrieval_ids, golden_context_ids, k=1):
    for golden_id in golden_context_ids:
        if golden_id in retrieval_ids[:k]:
            return 1
    return 0


def recall(retrieved_ids, expected_ids, k=1):
    retrieved_ids = retrieved_ids[:k]
    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)

    # 计算TP、FP和FN
    TP = len(retrieved_set & expected_set)  # 预测为正且实际为正的
    FP = len(retrieved_set - expected_set)  # 预测为正但实际为负的
    FN = len(expected_set - retrieved_set)  # 实际为正但预测为负的

    # 计算精确率（Precision）和召回率（Recall）
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return Recall


def precision(retrieved_ids, expected_ids, k=1):
    retrieved_ids = retrieved_ids[:k]
    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)

    # 计算TP、FP和FN
    TP = len(retrieved_set & expected_set)  # 预测为正且实际为正的
    FP = len(retrieved_set - expected_set)  # 预测为正但实际为负的
    FN = len(expected_set - retrieved_set)  # 实际为正但预测为负的

    # 计算精确率（Precision）和召回率（Recall）
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return Precision


text_qa_template_str = (
"Below are some examples:\n"
"Q: What is the capital of France? A: Paris.\n"
"Q: What is the boiling point of water in Celsius? A: 100°C.\n"
"---------------------\n"
"Below is the context information.\n"
"---------------------\n"
"{context_str}\n"
"---------------------\n"
"Based solely on the above context, and without using ANY prior knowledge, answer the following question as concisely as possible: {query_str} \n"
"Instructions:\n"
"*   Answer EXTREMELY briefly, focusing ONLY on the most essential information from the context.\n"
"*   Format your answer precisely as requested in the question if applicable e.g., \"X.X%\", \"YYYY-MM-DD\".\n"
"*   If the answer requires a calculation and the units are specified (e.g., percents), provide the result in those units, rounded as instructed."
)

text_qa_template = PromptTemplate(text_qa_template_str)

# Setup index query engine using LLM
# query_engine = index.as_query_engine(response_mode="compact")

query_engine.update_prompts({"response_synthesizer:text_qa_template": text_qa_template})
# query_engine = query_expansion([query_engine], query_number=4, similarity_top_k=10)
# query_engine = RetrieverQueryEngine.from_args(query_engine)


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

import collections
import os

categorized_evaluation_data = collections.defaultdict(lambda: {'scores': collections.defaultdict(float), 'count': 0})
def add_scores_by_category(category, new_scores):

    global categorized_evaluation_data
    category_data = categorized_evaluation_data[category]
    category_data['count'] += 1
    for metric, score in new_scores.items():
        category_data['scores'][metric] += score
def compute_and_format_categorized_averages():

    global categorized_evaluation_data
    final_output = "NLG Evaluation Metrics Averages by Category:\n"
    final_output += "=" * 40 + "\n"
    sorted_categories = sorted(categorized_evaluation_data.keys())
    if not sorted_categories:
        return "No evaluation data has been added yet."
    for category in sorted_categories:
        data = categorized_evaluation_data[category]
        scores = data['scores']
        count = data['count']
        final_output += f"--- Category: {category} (Evaluations: {count}) ---\n"
        if count == 0:
            final_output += "  No scores recorded for this category.\n"
        else:
            sorted_metrics = sorted(scores.keys())
            for metric in sorted_metrics:
                total_score = scores[metric]
                average_score = total_score / count
                final_output += f"  {metric}: {average_score:.4f}\n"
        final_output += "-" * 40 + "\n"
    return final_output.strip()
def write_categorized_averages_to_file(path):
    formatted_output = compute_and_format_categorized_averages()
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(formatted_output)
            f.write("\n\n")
        print(f"Categorized results successfully appended to {path}")
    except IOError as e:
        print(f"Error writing to file {path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# 添加分数到对应的类别
# add_scores_by_category('category_A', scores_1_cat_A)
# add_scores_by_category('category_B', scores_2_cat_B)
# add_scores_by_category('category_A', scores_3_cat_A)
# add_scores_by_category('category_C', scores_4_cat_C)
# average_results = compute_and_format_categorized_averages()
# print(average_results)

# 你也可以单独访问某个类别的数据（如果需要）
# print("\nRaw data for Category A:")
# print(categorized_evaluation_data['category_A']) 


for question, expected_answer, golden_context, golden_context_ids, question_type in zip(
        qa_dataset['question'],
        qa_dataset['answers'],
        qa_dataset['golden_sentences'],
        qa_dataset['golden_ids'],
        qa_dataset['question_types'],
):

    print(question)
    print(expected_answer)
    print(golden_context)
    print(golden_context_ids)
    print(question_type)
    # response = transform_and_query(question, cfg, query_engine)
    try:
        response = transform_and_query(question, cfg, query_engine)
        actual_response = response.response
        response = response.source_nodes
    except openai.APITimeoutError as e:
        print(f"OpenAI API 超时错误: {e}")
        continue
    except openai.APIError as e:
        print(f"OpenAI API 错误: {e}")
        continue
    except Exception as e:
        print(f"发生其他错误: {e}")
        continue
    # 返回node节点
    retrieval_ids = []
    retrieval_context = []
    response = sorted(response, key=lambda x: x.score, reverse=True)
    for source_node in response:
        retrieval_ids.append(source_node.metadata['id'])
        retrieval_context.append(source_node.get_content())

    print(retrieval_ids, "--- hzt ---", golden_context_ids)

    eval_result = evaluating_TRT(retrieval_ids, golden_context_ids)
    evaluateResults_TRT.add(eval_result)

    score = {"cos_1": hit(retrieval_ids, golden_context_ids, 1), "cos_3": hit(retrieval_ids, golden_context_ids, 3),
             "cos_5": hit(retrieval_ids, golden_context_ids, 5), "cos_10": hit(retrieval_ids, golden_context_ids, 10),
             "recall_1": recall(retrieval_ids, golden_context_ids, 1),
             "recall_3": recall(retrieval_ids, golden_context_ids, 3),
                "recall_5": recall(retrieval_ids, golden_context_ids, 5),
                "recall_10": recall(retrieval_ids, golden_context_ids, 10),
             "precision": precision(retrieval_ids, golden_context_ids, 1),
             "precision_3": precision(retrieval_ids, golden_context_ids, 3),
                "precision_5": precision(retrieval_ids, golden_context_ids, 5),
                "precision_10": precision(retrieval_ids, golden_context_ids, 10),
             "hit_2": hit(retrieval_ids, golden_context_ids, 2), "hit_4": hit(retrieval_ids, golden_context_ids, 4),
             "hit_8": hit(retrieval_ids, golden_context_ids, 8)}
    
    score.update(NLGEvaluate(actual_response, expected_answer))
    add_scores(score)
    print(compute_and_format_averages())
    all_num = all_num + 1

    print("TRT", '-' * 100)
    evaluateResults_TRT.print_results()
    print("TRT", '-' * 100)

    print("总数：" + str(all_num))
    add_scores_by_category(question_type, score)
path = "./0407-mathstral-7B-6606.txt"
evaluateResults_TRT.print_results_to_path(path, cfg, last_dir)
average_results = compute_and_format_categorized_averages()
print(average_results)
f = open(path, 'a')
formatted_output = "other:\n"
for metric, total_score in global_evaluation_scores.items():
    average_score = total_score / evaluation_count
    formatted_output += f"{metric}: {average_score:.4f}\n"
f.write(formatted_output)
f.close()
write_categorized_averages_to_file(path)
# python main.py --evaluateApiName="gpt-3.5-turbo" --evaluateApiKey="sk-FDMbb0bnifWopAgv4501226dEa4c45E5Ac81841123Eb80B0"
if __name__ == '__main__':
    print('Success')