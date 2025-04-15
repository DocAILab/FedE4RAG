# 因为本地evaluate与外部包重名，需要暂时删除路径
import os

from jury import Jury
import jury
import evaluate
from eval.evaluate_rag import EvaluationResult

NLG_EVALUATION_METRICS = [
    "chrf", "bleu", "meteor", "wer", "cer", "chrf_pp", "mauve", "perplexity",
    "rouge_rouge1", "rouge_rouge2", "rouge_rougeL", "rouge_rougeLsum"
]

ppl_bug_number=0
def evaluating_TGT(actual_responses, expect_answers):
    # omit_metrics = []
    # for metric in NLG_EVALUATION_METRICS:
    #     if metric not in metrics:
    #         omit_metrics.append(metric)

    # n = NLGEval(metrics_to_omit=omit_metrics)
    eval_result = EvaluationResult()
    references = []
    if type(expect_answers) == list:
        references = [str(response) for response in expect_answers]
    elif type(expect_answers) == str:
        references = [expect_answers]

    if type(actual_responses) == list:
        predictions = [str(response) for response in actual_responses]
    elif type(actual_responses) == str:
        predictions = [actual_responses]

    # Individual Metrics
    # scores = n.compute_individual_metrics(ref=reference, hyp=hypothesis)
    scorer = Jury(metrics=["chrf", "meteor", "rouge", "wer", "cer"])
    # chrf++
    chrf_plus = evaluate.load("chrf")
    score = chrf_plus.compute(predictions=predictions, references=[references], word_order=2)
    eval_result.results["chrf_pp"] = score["score"] / 100
    # perplexity:model id are needed
    perplexity = jury.load_metric("perplexity")
    score = perplexity.compute(predictions=predictions, references=references, model_id="openai-community/gpt2")
    eval_result.results["perplexity"] = score["mean_perplexity"]
    if int(eval_result.results["perplexity"]) > 1600:
        global ppl_bug_number
        ppl_bug_number = ppl_bug_number + 1
        print("\n\n" + "ppl_bug_number:" + str(ppl_bug_number) + "\n\n")
        eval_result.results["perplexity"] = 0
    score = scorer(predictions=predictions, references=[references])
    eval_result.results["chrf"] = score["chrf"]["score"]
    eval_result.results["meteor"] = score["meteor"]["score"]
    # 'rouge1': 0.6666666666666665, 'rouge2': 0.5714285714285715, 'rougeL': 0.6666666666666665, 'rougeLsum': 0.6666666666666665
    eval_result.results["rouge_rouge1"] = score["rouge"]["rouge1"]
    eval_result.results["rouge_rouge2"] = score["rouge"]["rouge2"]
    eval_result.results["rouge_rougeL"] = score["rouge"]["rougeL"]
    eval_result.results["rouge_rougeLsum"] = score["rouge"]["rougeLsum"]
    eval_result.results["wer"] = score["wer"]["score"]
    eval_result.results["cer"] = score["cer"]["score"]
    
    return eval_result
