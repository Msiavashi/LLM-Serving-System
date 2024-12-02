import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from torch.utils.data import DataLoader

def main():
    # Load model and tokenizer
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
    )

    # Enable KV cache
    model.config.use_cache = True

    # Prompts to feed into the model
    prompts = [
        "What is your favourite condiment?",
        "What are the benefits of a healthy diet?",
        "Explain the theory of relativity in simple terms.",
        "How do airplanes stay in the air?",
        "Describe the process of photosynthesis.",
        "What is the capital of France?",
        "How does blockchain technology work?",
        "What are the symptoms of the common cold?",
        "How do you make a perfect cup of coffee?",
        "Describe the life cycle of a butterfly.",
        "What are the key principles of democracy?",
        "How do you play the game of chess?",
        "Explain the process of human digestion.",
        "What is the role of the United Nations?",
        "What are the main functions of the human liver?",
        "What is the history of the Roman Empire?",
        "What are the benefits of learning a second language?",
        "What are the causes and effects of global warming?",
        "Describe the history and significance of the Eiffel Tower.",
        "What are the benefits of meditation?",
        "What are the different types of musical instruments?",
        "How does the human circulatory system function?",
        "What is the importance of biodiversity?",
        "Describe the process of making bread.",
        "What are the key elements of a successful business plan?",
        "How do you practice mindfulness?",
        "What are the different types of art movements?",
        "Explain the process of water purification.",
        "What are the benefits of recycling?",
        "How does the human nervous system work?",
        "What is the significance of the Statue of Liberty?",
        "What are the main causes of the French Revolution?"
    ]

    # Process prompts in batches
    batch_size = 32
    dataloader = DataLoader(prompts, batch_size=batch_size)

    total_decode_time = 0
    total_generated_tokens = 0
    
    for batch in dataloader:
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda")
        
        # Measure decode (model generation + decoding)
        decode_start = time.time()
        outputs = model.generate(inputs["input_ids"], max_new_tokens=1000, use_cache=True)
        decode_end = time.time()
        decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        total_decode_time += decode_end - decode_start
        
        # Subtract the length of the input tokens from the output tokens
        total_generated_tokens += sum(len(output) - len(input_ids) for output, input_ids in zip(outputs, inputs["input_ids"]))
        
        # Print results for the batch
        for prompt, response in zip(batch, decoded_outputs):
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print("-" * 80)

    # Report throughput metrics
    num_prompts = len(prompts)
    decode_throughput = total_generated_tokens / total_decode_time

    print(f"Processed {num_prompts} prompts.")
    print(f"Total Decode Time: {total_decode_time:.2f} seconds.")
    print(f"Decode Throughput: {decode_throughput:.2f} prompts per second.")

if __name__ == "__main__":
    main()
