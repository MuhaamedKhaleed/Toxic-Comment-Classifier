pip install transformers datasets torch scikit-learn


import zipfile
import os

# Specify the path to your zip file
zip_file_path = '/content/jigsaw-toxic-comment-classification-challenge (1).zip'

# Create a directory to extract the contents
extract_dir = '/content/jigsaw-toxic-comment-classification-challenge'
os.makedirs(extract_dir, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Files unzipped successfully!")



import pandas as pd

df = pd.read_csv('/content/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
print(df.head())
df_sample = df.sample(frac=0.05, random_state=42)  # Use 5% of the dataset

# Step 4: Sample a Smaller Dataset for Faster Training
df_sample = df.sample(frac=0.1, random_state=42)
df_sample['toxic'] = df_sample['toxic'].astype(int)  # Assuming 'toxic' column for toxic classification

from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_sample['comment_text'], df_sample['toxic'], test_size=0.2, random_state=42
)


from transformers import AutoTokenizer

model_name = 'distilbert-base-uncased'  # You can try smaller models like 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Adjust max_length for faster training
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=64)


import torch

class ToxicCommentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Convert labels to float
        item['labels'] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ToxicCommentDataset(train_encodings, train_labels.tolist())
test_dataset = ToxicCommentDataset(test_encodings, test_labels.tolist())

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)  # Use num_labels=6 for multi-label

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,  # Reduced epochs for quicker training
    per_device_train_batch_size=32,  # Increase batch size if memory allows
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,  # Accumulate gradients over two steps
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    fp16=True,  # Enable mixed precision training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Step 8: Evaluate the Model
results = trainer.evaluate()
print(results)

# Step 9: Save the Model
model.save_pretrained('./toxic_comment_model')
tokenizer.save_pretrained('./toxic_comment_model')

results = trainer.evaluate()
print(results)


# prompt: interface

from transformers import pipeline

# Load the saved model and tokenizer
model_path = './toxic_comment_model'
classifier = pipeline('text-classification', model=model_path, tokenizer=model_path)

def classify_comment(comment):
  result = classifier(comment)[0]
  return result

#@title Toxic Comment Classifier
comment_text = 'i hate u' #@param {type:"string"}
result = classify_comment(comment_text)

print(f"Comment: {comment_text}")
print(f"Label: {result['label']}")
print(f"Score: {result['score']}")


