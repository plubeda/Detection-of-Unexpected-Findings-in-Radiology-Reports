# Detection of Unexpected Findings in Radiology Reports


## Train model
Example of use:
```
python train.py --dataset_path /datasets/train.csv --batch_size 22 --learning_rate 1e-5 --max_epochs 30 --max_length 350 --lowercase --loss_function cross_entropy --optimizer_name adamw --output_dir ./output_models
```

## Evaluation
Example of use:
```
python test.py --dataset_path /datasets/test.csv --model_dir ./model_output --max_length 350 --lowercase --output_path ./prediction.txt
```

## Evaluation metrics
Example of use:
```
python evaluate_predictions.py --predictions_file ./prediction.txt --threshold 0.75
```
