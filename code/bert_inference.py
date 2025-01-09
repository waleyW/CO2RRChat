import os
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification
import argparse

def load_bert_model(model_path):
    """
    Load the trained BERT model and tokenizer
    
    Parameters:
    model_path (str): Path to the saved model
    
    Returns:
    tuple: (model, tokenizer, label_list)
    """
    # Load the model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    
    # Get the label list from the model configuration
    label_list = [f'Label_{i}' for i in range(model.config.num_labels)]
    
    return model, tokenizer, label_list

def bert_inference(model, tokenizer, query, label_list):
    """
    Perform text classification inference using the BERT model
    
    Parameters:
    model (BertForSequenceClassification): Trained model
    tokenizer (BertTokenizerFast): Tokenizer
    query (str): Text to be classified
    label_list (list): List of labels
    
    Returns:
    dict: Classification result
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Encode the input text
    inputs = tokenizer(
        query, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Disable gradient computation
    with torch.no_grad():
        # Get the model output
        outputs = model(**inputs)
        
        # Use softmax to get the probabilities for each class
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get the most likely label
        predicted_label_id = torch.argmax(probabilities, dim=-1).item()
        predicted_label = label_list[predicted_label_id]
        confidence = probabilities[0][predicted_label_id].item()
    
    return {
        "predicted_label": predicted_label,
        "label_id": predicted_label_id,
        "confidence": float(confidence),
        "all_probabilities": probabilities.tolist()[0]
    }

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained BERT model")
    parser.add_argument("--query", type=str, required=True, help="Text to be classified")
    args = parser.parse_args()
    
    # Load the model, tokenizer, and label list
    model, tokenizer, label_list = load_bert_model(args.model_path)
    
    # Perform inference on the given query
    result = bert_inference(model, tokenizer, args.query, label_list)
    
    # Print the classification results
    print("Classification Results:")
    print(f"Predicted Label: {result['predicted_label']}")
    print(f"Label ID: {result['label_id']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    print("Probabilities for each label:")
    for label, prob in zip(label_list, result['all_probabilities']):
        print(f"{label}: {prob:.4f}")

if __name__ == "__main__":
    main()