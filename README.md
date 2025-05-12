# FedE4RAG

This is the repository of the paper ***Privacy-Preserving Federal Embedding Learning for Localized Retrieval-Augmented Generation***.

FedE4RAG addresses data scarcity and privacy challenges in private RAG systems. It uses federated learning (FL) to collaboratively train client-side RAG retrieval models, keeping raw data localized. The framework employs knowledge distillation for effective server-client communication and homomorphic encryption to enhance parameter privacy. FedE4RAG aims to boost the performance of localized RAG retrievers by leveraging diverse client insights securely, balancing data utility and confidentiality, particularly demonstrated in sensitive domains like finance.

## Environment

#### Upstream Embedding Learning Environment

Run command below to install all the environment in need.

```
cd FedE
pip install -r requirements.txt
```

#### Downstream Question & Answer Environment

Create a Virtual Environment via conda (Recommended)：

```bash
conda create -n Fedrag-test python=3.11
conda install -r requirements
conda install openai==1.55.3
```

Install via pip：

```
pip install -r requirements
pip install openai==1.55.3
pip install jury --no-deps
```

## Data

We provide all datasets used in our experiments:

- The all datasets used are [DocAILab/FedE4RAG_Dataset · Datasets at Hugging Face](https://huggingface.co/datasets/DocAILab/FedE4RAG_Dataset).
- The datasets used for training are [train_data in DocAILab/FedE4RAG_Dataset](https://huggingface.co/datasets/DocAILab/FedE4RAG_Dataset/tree/main/train_data).

## Usage

### Upstream Embedding Learning

#### Step1	

Change the model training hyperparameters in the [FedE/main.py](https://github.com/DocAILab/FedE4RAG/blob/main/FedE/main.py).

#### Step2

Select the appropriate training data and copy it to the [FedE/select_data.json](https://github.com/DocAILab/FedE4RAG/blob/main/FedE/select_data.json).

#### Step3

Generate the fine-tuned model by executing the following shell script. (Before running, change the "data_path" augument in the script and code as needed)

```
cd ./FedE/
bash run.sh
```

### Downstream Question & Answer

- "The `bash.sh` and `bash1.sh` files provide scripts for directly evaluating your model.  You can use them by correctly filling in the path to your model within the scripts. The difference between them is that `bash1` additionally includes tests for the model's generation capabilities."
- "The `main_100_test.py`, `main_50_test.py`, and `response.py` are the specific evaluation files. You can customize the evaluation metrics and output files you need within them."

## Citation

```c
@misc{mao2025privacypreservingfederatedembeddinglearning,
      title={Privacy-Preserving Federated Embedding Learning for Localized Retrieval-Augmented Generation}, 
      author={Qianren Mao and Qili Zhang and Hanwen Hao and Zhentao Han and Runhua Xu and Weifeng Jiang and Qi Hu and Zhijun Chen and Tyler Zhou and Bo Li and Yangqiu Song and Jin Dong and Jianxin Li and Philip S. Yu},
      year={2025},
      eprint={2504.19101},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.19101}, 
}
```

## Acknowledgements

This project draws inspiration from and incorporates code elements of the FLGo project (https://github.com/WwZzz/easyFL).  We are grateful for the contributions and insights provided by the FLGo development team, which have been instrumental in advancing our project's development in the federated learning domain.
