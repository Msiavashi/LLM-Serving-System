import sys
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from tqdm import tqdm

def read_dataset(dataset_path, num_prompts):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    dataset = [(conv[0]["value"], conv[1]["value"]) 
                for data in dataset 
                if (conv := data.get("conversations", [])) and len(conv) >= 2]

    # Remove duplicates while maintaining order
    seen = set()
    ordered_dataset = []
    for item in dataset:
        if item not in seen:
            seen.add(item)
            ordered_dataset.append(item)

    # Select first `num_prompts` items
    filtered_dataset = [prompt for prompt, _ in ordered_dataset][:num_prompts]
    return filtered_dataset

def initialize_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        device_map='auto',
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        offload_folder=None,  # Prevent disk offloading
        low_cpu_mem_usage=True,
    )
    
    return model, tokenizer

def batch_inference(prompts, model, tokenizer, batch_size=64):
    all_outputs = []
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        torch.cuda.empty_cache()  # Clear cache before each batch
        
        batch_prompts = prompts[i:i + batch_size]
        
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                min_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1,
                use_cache=True
            )
        
        input_lengths = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, input_lengths:]
        decoded_outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        all_outputs.extend(decoded_outputs)
        
        # Clean up tensors
        # del inputs, outputs, new_tokens
        # torch.cuda.empty_cache()
    
    return all_outputs

def main():
    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer()
    
    # Read dataset
    dataset_path = "./examples/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    prompts = read_dataset(dataset_path, num_prompts=1000)
    
    # Run batch inference
    outputs = batch_inference(prompts, model, tokenizer)
    
    # Print results
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}\nGenerated: {output}\n{'='*50}\n")

if __name__ == "__main__":
    main()