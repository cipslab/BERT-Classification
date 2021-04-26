# BERT-Classification
Bert Classification for long documents

## Training
```python main.py --mode train --data_path PATH_TO_DATA --save_model --num_labels NUM_LABELS```

## Testing
```python main.py --mode test --data_path PATH_TO_DATA --num_labels NUM_LABELS --save_file PATH_TO_RESULTS```

## Data Format

story_id  | raw_text | label
------------- | ------------- | -------------
integer  | string | integer
