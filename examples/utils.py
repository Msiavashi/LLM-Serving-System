import json
import random
from tqdm.auto import tqdm


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

def generate_prompts(num_prompts, prompt_size, tokenizer):
    """ 
        FIXME: The sequence lengths are not 100% accurate, however, they are close enough for the purpose of examples.
    """
    prompts = []
    vocab = list(tokenizer.get_vocab().keys())
    
    with tqdm(total=num_prompts, desc="Generating Prompts") as pbar:
        for _ in range(num_prompts):
            prompt_tokens = random.choices(vocab, k=prompt_size)
            prompt_tokens = [int(tokenizer.convert_tokens_to_ids(token)) for token in prompt_tokens]
            prompt = tokenizer.decode(prompt_tokens)
            prompts.append(prompt)
            pbar.update(1)
    return prompts
