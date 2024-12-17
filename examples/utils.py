import json
from transformers import AutoTokenizer


def read_shared_gpt_dataset(dataset_path, num_prompts):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Filter and Transform in a Single Step
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

