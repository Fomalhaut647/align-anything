import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Any

def load_model_and_tokenizer(model_path: str, device: str = 'npu:0'):
    """Load model and tokenizer from path."""
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    device: str = 'npu:0'
) -> str:
    """Generate response for a given prompt."""
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def load_test_data(test_file: str) -> List[Dict[str, Any]]:
    """Load test data from parquet file and extract questions."""
    df = pd.read_parquet(test_file)
    # 只取前64个样本
    df = df.head(64)
    test_data = []
    for _, row in df.iterrows():
        test_data.append({
            'prompt': row['question'],
        })
    return test_data

def evaluate_models(
    base_model_path: str,
    dpo_model_path: str,
    test_file: str,
    output_dir: str,
    device: str = 'npu:0',
    batch_size: int = 4
):
    """Evaluate both models on test data and save results."""
    # Load models
    print("Loading base model...")
    base_model, base_tokenizer = load_model_and_tokenizer(base_model_path, device)
    print("Loading DPO model...")
    dpo_model, dpo_tokenizer = load_model_and_tokenizer(dpo_model_path, device)

    # Load test data
    test_data = load_test_data(test_file)
    
    results = []
    
    # Evaluate each test case in batches
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        batch = test_data[i:i + batch_size]
        batch_prompts = [item['prompt'] for item in batch]
        
        # Generate responses for batch
        base_responses = []
        dpo_responses = []
        
        for prompt in batch_prompts:
            base_response = generate_response(base_model, base_tokenizer, prompt, device=device)
            dpo_response = generate_response(dpo_model, dpo_tokenizer, prompt, device=device)
            base_responses.append(base_response)
            dpo_responses.append(dpo_response)
        
        # Save results for batch
        for j, item in enumerate(batch):
            result = {
                'prompt': item['prompt'],
                'base_response': base_responses[j],
                'dpo_response': dpo_responses[j],
            }
            results.append(result)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Create a summary DataFrame
    df = pd.DataFrame(results)
    summary_file = os.path.join(output_dir, 'evaluation_summary.csv')
    df.to_csv(summary_file, index=False)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    # Model paths
    BASE_MODEL_PATH = "data/Qwen2.5-0.5B-Instruct"
    DPO_MODEL_PATH = "outputs/qwen_2_5_dpo/slice_end"
    
    # Test data path
    TEST_FILE = "data/align_anything_t2t/val_1k.parquet"
    
    # Output directory
    OUTPUT_DIR = "scripts/qwen2_5/evaluation_results"
    
    # Run evaluation
    evaluate_models(
        base_model_path=BASE_MODEL_PATH,
        dpo_model_path=DPO_MODEL_PATH,
        test_file=TEST_FILE,
        output_dir=OUTPUT_DIR
    ) 