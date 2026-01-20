# Fine-tuning OpenAI Models for NDA Document Understanding

This guide explains how to fine-tune OpenAI models to better understand and answer questions about NDA documents.

## Prerequisites

1. An OpenAI API key with fine-tuning permissions
2. Python 3.8+
3. Required packages (see `requirements.txt`)

## Setup

1. Make sure your OpenAI API key is set in your `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Fine-tuning Process

### 1. Prepare Training Data

The script `openai_finetune.py` will convert your CSV datasets into the JSONL format required by OpenAI for fine-tuning. 

The dataset CSV files should have the following columns:
- `sentence1`: The NDA clause
- `sentence2`: A statement about the clause
- `label`: The relationship between the clause and statement (0=contradiction, 1=entailment, 2=neutral)
- `gold_label`: Text label (optional)

### 2. Run the Fine-tuning Script

```
python openai_finetune.py
```

This will:
1. Process your training and validation datasets
2. Create JSONL files with properly formatted examples
3. Upload the files to OpenAI
4. Start a fine-tuning job
5. Check the initial status of the job

### 3. Monitor Fine-tuning Progress

To check the status of your fine-tuning job:

```
python openai_finetune.py check_status YOUR_JOB_ID
```

To list all your fine-tuning jobs:

```
python openai_finetune.py list_jobs
```

Fine-tuning typically takes 1-4 hours depending on the dataset size.

### 4. Test Your Fine-tuned Model

Once fine-tuning is complete, you'll receive a fine-tuned model ID. You can test it with:

```
python openai_finetune.py test_model YOUR_MODEL_ID
```

This will compare the responses of your fine-tuned model against the standard GPT-3.5 Turbo model on several test examples.

### 5. Evaluate Model Accuracy

To evaluate the accuracy of your fine-tuned model and compare it with the standard model:

```
python openai_finetune.py evaluate YOUR_MODEL_ID
```

This will:
1. Evaluate the model on a sample of test cases (default: 100 samples)
2. Calculate and display the following metrics:
   - Overall accuracy for both fine-tuned and standard models
   - Classification reports with precision, recall, and F1-score
   - Confusion matrices
3. Save detailed results to a JSON file
4. Generate and save confusion matrix visualizations

The evaluation provides:
- **Accuracy**: The percentage of correctly classified examples
- **Precision**: How many of the predicted positives are actually positive
- **Recall**: How many of the actual positives are correctly predicted
- **F1-score**: The harmonic mean of precision and recall
- **Improvement**: The percentage improvement of the fine-tuned model over the standard model

Example output:
```
================================================================================
ACCURACY EVALUATION RESULTS
================================================================================
Number of test samples: 100
Standard model accuracy: 0.7400
Fine-tuned model accuracy: 0.8900
Improvement: 15.00%

Standard Model Classification Report:
              precision    recall  f1-score   support
contradiction       0.71      0.63      0.67        24
   entailment       0.75      0.85      0.80        47
      neutral       0.74      0.66      0.70        29
    accuracy                           0.74       100
   macro avg       0.73      0.71      0.72       100
weighted avg       0.74      0.74      0.74       100

Fine-tuned Model Classification Report:
              precision    recall  f1-score   support
contradiction       0.92      0.92      0.92        24
   entailment       0.88      0.94      0.91        47
      neutral       0.88      0.79      0.84        29
    accuracy                           0.89       100
   macro avg       0.89      0.88      0.89       100
weighted avg       0.89      0.89      0.89       100

Detailed results saved to accuracy_results_ft_gpt-3.5-turbo_your-org_custom-suffix_id.json
Confusion matrix visualization saved to confusion_matrix_ft_gpt-3.5-turbo_your-org_custom-suffix_id.png
```

## Integrating with Your Application

To use your fine-tuned model in your existing application:

1. Update the `use_finetuned_model.py` script with your fine-tuned model ID
2. Run the example:
   ```
   python use_finetuned_model.py
   ```

3. To integrate with your application:
   - Use the `EnhancedDocumentProcessor` class instead of the regular `DocumentProcessor`
   - Pass your fine-tuned model ID when initializing the class

Example:
```python
from use_finetuned_model import EnhancedDocumentProcessor

# Initialize with your fine-tuned model ID
processor = EnhancedDocumentProcessor("ft:gpt-3.5-turbo:your-org:custom_suffix:id")

# Use as you would the regular DocumentProcessor
processor.process_document("your_document.pdf", "user_id")
answer = processor.get_answer("What are the confidentiality obligations?", "user_id")
```

## Benefits of Fine-tuning

1. **Domain-specific understanding**: The model will better understand legal terminology and NDA clauses
2. **Improved accuracy**: Fine-tuned models can more accurately determine relationships between NDA clauses and statements
3. **More relevant answers**: Responses will be more focused on the legal implications of NDA documents
4. **Format consistency**: The model maintains the formatting style of your documents

## Important Notes

1. Fine-tuning currently works with `gpt-3.5-turbo`, not with GPT-4 models
2. Fine-tuning costs depend on the number of tokens in your dataset and the training epochs
3. The model will still have token limitations (4K tokens for gpt-3.5-turbo)
4. For best results, use a diverse set of high-quality examples in your training data

## Troubleshooting

- **File upload errors**: Ensure your JSONL files are properly formatted
- **Training errors**: Check that your examples have appropriate length (not too short or too long)
- **API errors**: Verify your API key has fine-tuning permissions
- **Rate limits**: If you encounter rate limits, space out your API requests