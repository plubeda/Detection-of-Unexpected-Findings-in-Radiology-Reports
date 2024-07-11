import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import argparse

def train_model(dataset_path, batch_size, learning_rate, max_epochs, max_length, lowercase, loss_function, optimizer_name, output_dir):
    # Load the dataset
    dataset = load_dataset('csv', data_files=dataset_path)
    
    # Preprocess the dataset
    def preprocess_function(examples):
        tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-biomedical-clinical-es')
        if lowercase:
            examples['text'] = [text.lower() for text in examples['text']]
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    # Set up the model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')

    # Define the optimizer
    optimizer = {
        'adamw': torch.optim.AdamW,
    }.get(optimizer_name, torch.optim.AdamW)(model.parameters(), lr=learning_rate)
    
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,  # Save directory
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=max_epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        save_total_limit=2,  # Limit the total amount of checkpoints
        save_steps=10_000,  # Save every 10,000 steps
    )

    # Set up the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        optimizers=(optimizer, None),  # optimizer and scheduler (set to None)
    )

    # Train the model
    trainer.train()
    
    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RoBERTa model.")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file.')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for training.')
    parser.add_argument('--max_epochs', type=int, required=True, help='Maximum number of epochs for training.')
    parser.add_argument('--max_length', type=int, required=True, help='Maximum sequence length for the tokenizer.')
    parser.add_argument('--lowercase', action='store_true', help='Whether to lowercase the input text.')
    parser.add_argument('--loss_function', type=str, required=True, choices=['cross_entropy', 'mse'], help='Loss function to use.')
    parser.add_argument('--optimizer_name', type=str, required=True, choices=['adamw', 'sgd'], help='Optimizer to use.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the model.')

    args = parser.parse_args()
    
    train_model(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        max_length=args.max_length,
        lowercase=args.lowercase,
        loss_function=args.loss_function,
        optimizer_name=args.optimizer_name,
        output_dir=args.output_dir
    )

