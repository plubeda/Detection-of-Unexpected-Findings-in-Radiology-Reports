import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset
import argparse
import numpy as np

def test_model(dataset_path, model_dir, max_length, lowercase, output_path):
    # Load the dataset
    dataset = load_dataset('csv', data_files=dataset_path)
    
    # Preprocess the dataset
    def preprocess_function(examples):
        tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        if lowercase:
            examples['text'] = [text.lower() for text in examples['text']]
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Load the model
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Perform predictions
    predictions = []
    probabilities = []

    for batch in tokenized_datasets['test']:
        with torch.no_grad():
            inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in batch.items() if k in tokenizer.model_input_names}
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).numpy()
            pred = np.argmax(probs, axis=1).item()
            predictions.append(pred)
            probabilities.append(probs.tolist())

    # Save predictions and probabilities to a file
    with open(output_path, 'w') as f:
        for prediction, probability in zip(predictions, probabilities):
            f.write(f"Prediction: {prediction}, Probabilities: {probability}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a RoBERTa model.")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file.')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory where the model is saved.')
    parser.add_argument('--max_length', type=int, required=True, help='Maximum sequence length for the tokenizer.')
    parser.add_argument('--lowercase', action='store_true', help='Whether to lowercase the input text.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the predictions.')

    args = parser.parse_args()
    
    test_model(
        dataset_path=args.dataset_path,
        model_dir=args.model_dir,
        max_length=args.max_length,
        lowercase=args.lowercase,
        output_path=args.output_path
    )

