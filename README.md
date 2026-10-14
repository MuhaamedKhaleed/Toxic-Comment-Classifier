# Toxic Comment Classifier

An end-to-end NLP pipeline for detecting toxic comments using fine-tuned transformer architectures (`distilbert-base-uncased`). Built with PyTorch and Hugging Face Transformers, this project demonstrates data preprocessing, tokenization, transfer learning, mixed-precision training, and model inference on the Jigsaw Toxic Comment Classification dataset.

## Features

- **Transformer-Based Architecture:** Leverages pre-trained `distilbert-base-uncased` for lightweight and low-latency text classification.
- **Optimized Training:** Utilizes Hugging Face `Trainer` with mixed precision (`fp16`) and gradient accumulation for fast and memory-efficient fine-tuning.
- **Custom PyTorch Dataset:** Implements an optimized `torch.utils.data.Dataset` interface for structured encoding and label ingestion.
- **Simple Inference Pipeline:** Includes a ready-to-use Hugging Face `pipeline` wrapper for real-time inference on arbitrary user text.

## Tech Stack

- **Deep Learning Framework:** PyTorch
- **NLP & Model Hub:** Hugging Face Transformers, Datasets, Tokenizers
- **Data Manipulation & Metrics:** Pandas, Scikit-Learn
- **Acceleration:** Hugging Face Accelerate

## Project Structure

text
├── train.py                  # Training and evaluation script
├── toxic_comment_model/      # Exported model weights and tokenizer
├── results/                  # Checkpoints from Trainer
├── logs/                     # Training run logs
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation


## Setup and Installation

### 1. Clone the Repository
bash
git clone https://github.com/your-username/Toxic-Comment-Classifier.git
cd Toxic-Comment-Classifier


### 2. Configure Environment
bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt


### 3. Data Preparation
Download the Jigsaw Toxic Comment Classification dataset from Kaggle and place the archive in your working directory:
bash
# Expected dataset archive
jigsaw-toxic-comment-classification-challenge.zip


### 4. Training
Run the fine-tuning pipeline:
bash
python train.py


### 5. Inference
Use the saved model artifacts with the Hugging Face pipeline:
python
from transformers import pipeline

classifier = pipeline('text-classification', model='./toxic_comment_model', tokenizer='./toxic_comment_model')
result = classifier("This is an exemplary comment.")
print(result)
