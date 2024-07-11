import argparse
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, classification_report

def evaluate_predictions(predictions_file, threshold):
    true_labels = []
    predicted_labels = []
    probabilities = []

    # Read the predictions and probabilities from the file
    with open(predictions_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            true_label = int(parts[0].split(':')[1].strip())
            prob = float(parts[1].split(':')[1].strip().replace('[', '').replace(']', '').split()[1])
            
            true_labels.append(true_label)
            probabilities.append(prob)
            predicted_labels.append(1 if prob >= threshold else 0)
    
    # Calculate overall metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, probabilities)

    # Calculate per-class metrics
    report = classification_report(true_labels, predicted_labels, target_names=['control', 'unexpected finding'], output_dict=True)
    
    # Print overall metrics
    print(f'Overall Metrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'AUC: {auc:.4f}')
   
    # Print per-class metrics
    print(f'\nPer-Class Metrics:')
    for label, metrics in report.items():
        if label in ['control', 'unexpected finding']:
            print(f"\nClass '{label}':")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1-score']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions using a threshold.")
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to the file containing predictions and probabilities.')
    parser.add_argument('--threshold', type=float, required=True, help='Probability threshold to classify as 1 or 0.')

    args = parser.parse_args()
    
    evaluate_predictions(args.predictions_file, args.threshold)
