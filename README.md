# FedE4RAG



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

