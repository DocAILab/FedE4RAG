# FedE4RAG

This is the repository of the paper ***Privacy-Preserving Federal Embedding Learning for Localized Retrieval-Augmented Generation***.

FedE4RAG addresses data scarcity and privacy challenges in private RAG systems. It uses federated learning (FL) to collaboratively train client-side RAG retrieval models, keeping raw data localized. The framework employs knowledge distillation for effective server-client communication and homomorphic encryption to enhance parameter privacy. FedE4RAG aims to boost the performance of localized RAG retrievers by leveraging diverse client insights securely, balancing data utility and confidentiality, particularly demonstrated in sensitive domains like finance.

## Environment

Run command below to install all the environment in need.

```
pip install -r requirements.txt
```

## Data

We provide all datasets used in our experiments:

- The all datasets used are [DocAILab/FedE4RAG_Dataset · Datasets at Hugging Face](https://huggingface.co/datasets/DocAILab/FedE4RAG_Dataset).
- The datasets used for training are [train_data in DocAILab/FedE4RAG_Dataset](https://huggingface.co/datasets/DocAILab/FedE4RAG_Dataset/tree/main/train_data).

## Usage

### Upstream Embedding Learning:

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

### Downstream Question & Answer:



## Citation

```c

```

