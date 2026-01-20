import pandas as pd
import json
import openai
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set it in your .env file.")

def prepare_finetune_data(csv_paths, output_path, max_samples=None):
    """
    Prepare training data for fine-tuning from multiple CSV files.
    
    Args:
        csv_paths: List of paths to CSV files
        output_path: Path to save the JSONL file
        max_samples: Maximum number of samples to use from each file (for testing)
    
    Returns:
        Path to the created JSONL file
    """
    all_data = []
    
    # Process each CSV file
    for csv_path in csv_paths:
        print(f"Processing {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Limit samples if specified
        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, random_state=42)
        
        # Create messages for each example
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Map numeric labels to text
            label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
            label_text = label_map.get(int(row['label']), "")
            gold_label = row.get('gold_label', label_text)
            
            # Clean and truncate the text to avoid token limits
            sentence1 = str(row['sentence1']).strip()
            sentence2 = str(row['sentence2']).strip()
            
            # Create conversation format
            messages = [
                {"role": "system", "content": "You are an expert in legal documents, especially NDAs (Non-Disclosure Agreements). Help users understand the relationships between NDA clauses and statements."},
                {"role": "user", "content": f"NDA Clause: {sentence1}\n\nStatement: {sentence2}\n\nDoes the NDA clause entail, contradict, or is neutral to the statement?"},
                {"role": "assistant", "content": f"The NDA clause {gold_label}s the statement. This means that {explain_relationship(gold_label, sentence1, sentence2)}"}
            ]
            
            all_data.append({"messages": messages})
    
    # Write to JSONL file
    with open(output_path, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created fine-tuning dataset with {len(all_data)} examples at {output_path}")
    return output_path

def explain_relationship(relationship, clause, statement):
    """Generate an explanation based on the relationship type"""
    if relationship == "entailment" or relationship == "entail":
        return "the NDA clause logically implies or supports the statement."
    elif relationship == "contradiction" or relationship == "contradict":
        return "the NDA clause directly contradicts or negates the statement."
    else:  # neutral
        return "the NDA clause neither directly supports nor contradicts the statement."

def start_finetune_job(training_file_path, validation_file_path=None):
    """
    Start a fine-tuning job with OpenAI.
    
    Args:
        training_file_path: Path to the training data JSONL file
        validation_file_path: Optional path to validation data JSONL file
    
    Returns:
        Fine-tuning job ID
    """
    print("Uploading training file...")
    training_file = openai.files.create(
        file=open(training_file_path, "rb"),
        purpose="fine-tune"
    )
    
    validation_file_id = None
    if validation_file_path:
        print("Uploading validation file...")
        validation_file = openai.files.create(
            file=open(validation_file_path, "rb"),
            purpose="fine-tune"
        )
        validation_file_id = validation_file.id
    
    # Wait for files to be processed
    print("Waiting for files to be processed...")
    time.sleep(10)
    
    # Create fine-tuning job
    job_params = {
        "training_file": training_file.id,
        "model": "gpt-3.5-turbo",  # Currently, gpt-3.5-turbo is the model available for fine-tuning
    }
    
    if validation_file_id:
        job_params["validation_file"] = validation_file_id
    
    print("Creating fine-tuning job...")
    try:
        finetune_job = openai.fine_tuning.jobs.create(**job_params)
        print(f"Fine-tuning job created: {finetune_job.id}")
        return finetune_job.id
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        return None

def check_finetune_status(job_id):
    """Check the status of a fine-tuning job"""
    try:
        job = openai.fine_tuning.jobs.retrieve(job_id)
        print(f"Job {job_id} status: {job.status}")
        
        if hasattr(job, 'fine_tuned_model') and job.fine_tuned_model:
            print(f"Fine-tuned model ID: {job.fine_tuned_model}")
        
        return job
    except Exception as e:
        print(f"Error retrieving job status: {e}")
        return None

def list_finetune_jobs():
    """List all fine-tuning jobs"""
    try:
        jobs = openai.fine_tuning.jobs.list()
        print("Recent fine-tuning jobs:")
        for job in jobs.data:
            print(f"ID: {job.id}, Model: {job.model}, Status: {job.status}")
        return jobs
    except Exception as e:
        print(f"Error listing jobs: {e}")
        return None

def test_finetuned_model(model_id, test_file_path, num_samples=5):
    """
    Test a fine-tuned model on some examples from the test set.
    
    Args:
        model_id: ID of the fine-tuned model
        test_file_path: Path to test CSV file
        num_samples: Number of samples to test
    """
    print(f"Testing fine-tuned model {model_id} on {num_samples} samples...")
    
    # Load test data
    df = pd.read_csv(test_file_path)
    if len(df) > num_samples:
        test_samples = df.sample(num_samples, random_state=42)
    else:
        test_samples = df
    
    for _, row in test_samples.iterrows():
        sentence1 = str(row['sentence1']).strip()
        sentence2 = str(row['sentence2']).strip()
        true_label = row.get('gold_label', "")
        
        prompt = f"NDA Clause: {sentence1}\n\nStatement: {sentence2}\n\nDoes the NDA clause entail, contradict, or is neutral to the statement?"
        
        try:
            # Standard model for comparison
            standard_response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            # Fine-tuned model
            finetuned_response = openai.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are an expert in legal documents, especially NDAs (Non-Disclosure Agreements). Help users understand the relationships between NDA clauses and statements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            print("\n" + "="*80)
            print(f"CLAUSE: {sentence1[:150]}...")
            print(f"STATEMENT: {sentence2}")
            print(f"TRUE RELATIONSHIP: {true_label}")
            print("\nSTANDARD MODEL RESPONSE:")
            print(standard_response.choices[0].message.content)
            print("\nFINE-TUNED MODEL RESPONSE:")
            print(finetuned_response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error testing model: {e}")

def evaluate_model_accuracy(model_id, test_file_path, num_samples=100, save_results=True):
    """
    Evaluate the accuracy of a fine-tuned model on the test set.
    
    Args:
        model_id: ID of the fine-tuned model
        test_file_path: Path to test CSV file
        num_samples: Number of samples to evaluate
        save_results: Whether to save results to a file
    
    Returns:
        Dictionary containing accuracy metrics
    """
    print(f"Evaluating fine-tuned model {model_id} on {num_samples} samples...")
    
    # Load test data
    df = pd.read_csv(test_file_path)
    if len(df) > num_samples:
        test_samples = df.sample(num_samples, random_state=42)
    else:
        test_samples = df
    
    # For storing results
    results = {
        "finetuned_preds": [],
        "standard_preds": [],
        "true_labels": [],
        "finetuned_correct": 0,
        "standard_correct": 0,
        "total_samples": len(test_samples),
        "detailed_results": []
    }
    
    # Label mapping for evaluation
    label_mapping = {
        "contradiction": 0,
        "entailment": 1, 
        "neutral": 2
    }
    
    # Process each test example
    for _, row in tqdm(test_samples.iterrows(), total=len(test_samples), desc="Evaluating"):
        sentence1 = str(row['sentence1']).strip()
        sentence2 = str(row['sentence2']).strip()
        
        # Get true label
        if 'gold_label' in row and row['gold_label']:
            true_label_text = row['gold_label']
        else:
            label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
            true_label_text = label_map.get(int(row['label']), "neutral")
            
        true_label_idx = label_mapping.get(true_label_text, 2)  # Default to neutral if label is unknown
        results["true_labels"].append(true_label_idx)
        
        prompt = f"NDA Clause: {sentence1}\n\nStatement: {sentence2}\n\nDoes the NDA clause entail, contradict, or is neutral to the statement? Answer with only one word: 'entailment', 'contradiction', or 'neutral'."
        
        try:
            # Standard model prediction
            standard_response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            # Fine-tuned model prediction
            finetuned_response = openai.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are an expert in legal documents, especially NDAs (Non-Disclosure Agreements). Help users understand the relationships between NDA clauses and statements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            # Extract predictions
            standard_text = standard_response.choices[0].message.content.lower()
            finetuned_text = finetuned_response.choices[0].message.content.lower()
            
            # Parse model responses to get the label
            standard_label = None
            finetuned_label = None
            
            for label in ["contradiction", "entailment", "neutral"]:
                if label in standard_text:
                    standard_label = label
                if label in finetuned_text:
                    finetuned_label = label
            
            # If no label was found, use a fallback
            if standard_label is None:
                standard_label = "neutral"
            if finetuned_label is None:
                finetuned_label = "neutral"
                
            standard_label_idx = label_mapping.get(standard_label, 2)
            finetuned_label_idx = label_mapping.get(finetuned_label, 2)
            
            results["standard_preds"].append(standard_label_idx)
            results["finetuned_preds"].append(finetuned_label_idx)
            
            # Check if predictions are correct
            if standard_label_idx == true_label_idx:
                results["standard_correct"] += 1
            if finetuned_label_idx == true_label_idx:
                results["finetuned_correct"] += 1
                
            # Store detailed result for this example
            results["detailed_results"].append({
                "clause": sentence1[:150] + "...",
                "statement": sentence2,
                "true_label": true_label_text,
                "standard_pred": standard_label,
                "finetuned_pred": finetuned_label,
                "standard_correct": standard_label_idx == true_label_idx,
                "finetuned_correct": finetuned_label_idx == true_label_idx
            })
            
        except Exception as e:
            print(f"Error evaluating example: {e}")
    
    # Calculate accuracy
    results["standard_accuracy"] = results["standard_correct"] / results["total_samples"]
    results["finetuned_accuracy"] = results["finetuned_correct"] / results["total_samples"]
    
    # Calculate classification metrics
    results["finetuned_report"] = classification_report(
        results["true_labels"], 
        results["finetuned_preds"],
        target_names=["contradiction", "entailment", "neutral"],
        output_dict=True
    )
    
    results["standard_report"] = classification_report(
        results["true_labels"], 
        results["standard_preds"],
        target_names=["contradiction", "entailment", "neutral"],
        output_dict=True
    )
    
    # Generate confusion matrices
    results["finetuned_cm"] = confusion_matrix(
        results["true_labels"], 
        results["finetuned_preds"],
        labels=[0, 1, 2]
    )
    
    results["standard_cm"] = confusion_matrix(
        results["true_labels"], 
        results["standard_preds"],
        labels=[0, 1, 2]
    )
    
    # Print results
    print("\n" + "="*80)
    print("ACCURACY EVALUATION RESULTS")
    print("="*80)
    print(f"Number of test samples: {results['total_samples']}")
    print(f"Standard model accuracy: {results['standard_accuracy']:.4f}")
    print(f"Fine-tuned model accuracy: {results['finetuned_accuracy']:.4f}")
    print(f"Improvement: {(results['finetuned_accuracy'] - results['standard_accuracy'])*100:.2f}%")
    
    print("\nStandard Model Classification Report:")
    print(classification_report(
        results["true_labels"], 
        results["standard_preds"],
        target_names=["contradiction", "entailment", "neutral"]
    ))
    
    print("\nFine-tuned Model Classification Report:")
    print(classification_report(
        results["true_labels"], 
        results["finetuned_preds"],
        target_names=["contradiction", "entailment", "neutral"]
    ))
    
    # Save results if requested
    if save_results:
        result_file = f"accuracy_results_{model_id.replace(':', '_')}.json"
        with open(result_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_json = results.copy()
            results_json["finetuned_cm"] = results["finetuned_cm"].tolist()
            results_json["standard_cm"] = results["standard_cm"].tolist()
            json.dump(results_json, f, indent=2)
        print(f"\nDetailed results saved to {result_file}")
        
        # Generate and save confusion matrix visualizations
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(results["standard_cm"], annot=True, fmt='d', cmap='Blues',
                   xticklabels=["contradiction", "entailment", "neutral"],
                   yticklabels=["contradiction", "entailment", "neutral"])
        plt.title('Standard Model Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(results["finetuned_cm"], annot=True, fmt='d', cmap='Blues',
                   xticklabels=["contradiction", "entailment", "neutral"],
                   yticklabels=["contradiction", "entailment", "neutral"])
        plt.title('Fine-tuned Model Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{model_id.replace(':', '_')}.png")
        print(f"Confusion matrix visualization saved to confusion_matrix_{model_id.replace(':', '_')}.png")
    
    return results

def main():
    # Define file paths
    train_path = "datasets/train-00000-of-00001-2.csv"
    dev_path = "datasets/dev-00000-of-00001-2.csv"
    test_path = "datasets/test-00000-of-00001.csv"
    
    # Prepare data for fine-tuning
    train_jsonl = "nda_train_finetune.jsonl"
    val_jsonl = "nda_val_finetune.jsonl"
    
    # For demonstration, use a smaller subset
    # Change max_samples to None to use the entire dataset
    max_samples = 200  # Limit samples for demonstration
    
    # Create training and validation data
    prepare_finetune_data([train_path], train_jsonl, max_samples)
    prepare_finetune_data([dev_path], val_jsonl, max_samples//5)
    
    # Start fine-tuning job
    job_id = start_finetune_job(train_jsonl, val_jsonl)
    
    if job_id:
        print(f"Fine-tuning job started with ID: {job_id}")
        print("You can check the status of your fine-tuning job by running:")
        print(f"python openai_finetune.py check_status {job_id}")
        
        # Initial status check
        check_finetune_status(job_id)
    else:
        print("Failed to start fine-tuning job.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == "list_jobs":
        list_finetune_jobs()
    elif sys.argv[1] == "check_status" and len(sys.argv) == 3:
        check_finetune_status(sys.argv[2])
    elif sys.argv[1] == "test_model" and len(sys.argv) == 3:
        test_finetuned_model(sys.argv[2], "datasets/test-00000-of-00001.csv")
    elif sys.argv[1] == "evaluate" and len(sys.argv) == 3:
        evaluate_model_accuracy(sys.argv[2], "datasets/test-00000-of-00001.csv")
    else:
        print("Usage:")
        print("  python openai_finetune.py                 # Run the full fine-tuning pipeline")
        print("  python openai_finetune.py list_jobs       # List all fine-tuning jobs")
        print("  python openai_finetune.py check_status JOB_ID   # Check status of a job")
        print("  python openai_finetune.py test_model MODEL_ID   # Test a fine-tuned model")
        print("  python openai_finetune.py evaluate MODEL_ID     # Evaluate model accuracy")