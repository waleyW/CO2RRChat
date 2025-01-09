import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments  # Changed from Roberta to Bert
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import argparse
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



class CO2RRDataset(Dataset):
    def __init__(self, questions, labels, tokenizer, max_len=512):
        self.questions = questions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):
        question = self.questions[index]
        label = self.labels[index]
        

        encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': label
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {'accuracy': accuracy}

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.base_model
    data_path = args.material
    output_dir = args.output_folder
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    random_state=args.random_state
    test_size=args.test_size
    

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    try:
      
        df = pd.read_csv(args.material, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df = pd.read_csv(args.material, encoding='GBK')
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['Label'])
    
    train_df.to_csv(os.path.join(output_dir, 'train_dataset.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_dataset.csv'), index=False)
    
    

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    train_dataset = CO2RRDataset(train_df['Question'].tolist(), train_df['Label'].tolist(), tokenizer)
    val_dataset = CO2RRDataset(val_df['Question'].tolist(), val_df['Label'].tolist(), tokenizer)


    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=args.weight_decay,
        logging_dir='./logs',
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        lr_scheduler_type=args.lr_scheduler_type,  
        warmup_ratio=args.warmup_ratio  
    )



    model = BertForSequenceClassification.from_pretrained(args.base_model, num_labels=len(df['Label'].unique()))
    model.to(device)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()

    print(f"Validation results: {eval_results}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    predictions = trainer.predict(val_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=-1)
    
    results_df = pd.DataFrame({
        'Question': val_df['Question'].tolist(),
        'True Label': val_df['Label'].tolist(),
        'Predicted Label': predicted_labels
    })
    results_file_path = os.path.join(output_dir, 'best_epoch_validation_results.csv')
    results_df.to_csv(results_file_path, index=False)
    print(f"Results from best model epoch saved to {results_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--material", type=str, required=True, help="Path to the input CSV file") 
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, required=True, help="Number of batch size")
    #parser.add_argument("--warmup_steps", type=int, required=True, help="Number of warmup_steps")
    parser.add_argument("--warmup_ratio", type=float, required=True, help="Number of warmup_steps")
    parser.add_argument("--random_state", type=int, required=True, help="Number of random_state")
    parser.add_argument("--weight_decay", type=float, required=True, help="Weight decay for optimizer")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for training")
    parser.add_argument("--test_size", type=float, required=True, help="Proportion of the dataset to include in the validation split")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the fine-tuned model")
    parser.add_argument("--lr_scheduler_type", type=str, choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'], default='linear', help="Type of learning rate scheduler to use")
    args = parser.parse_args()
    main(args)