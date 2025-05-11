import time
import numpy as np
import json
from text_generation import InferenceClient

# A simple dataset loader (modify if you have a shared utils function)
def read_shared_gpt_dataset(path, num_prompts):
    with open(path, 'r') as f:
        data = json.load(f)
    return data[:num_prompts]

def main():
    dataset_path = "./examples/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    num_prompts = 64
    prompts = read_shared_gpt_dataset(dataset_path, num_prompts)
    
    api_url = "http://localhost:8000"  # adjust if needed
    client = InferenceClient(api_url)

    durations = []
    total_start = time.time()
    
    for prompt in prompts:
        start = time.time()
        response = client.generate(prompt, max_new_tokens=50, temperature=0.7)
        # If response is streaming, ensure full text is generated (e.g., response.text)
        end = time.time()
        durations.append(end - start)
    
    total_end = time.time()
    avg_duration = np.mean(durations)
    throughput = num_prompts / (total_end - total_start)
    
    print(f"Average job completion time: {avg_duration:.4f} seconds")
    print(f"Throughput: {throughput:.2f} jobs per second")

if __name__ == "__main__":
    main()
