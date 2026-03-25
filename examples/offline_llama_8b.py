import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engines.factory import EngineFactory
from src.schedulers.factory import SchedulerFactory
from src.models.model_factory import ModelFactory
from src.schedulers.utils import CleanOutputFormatter

def usage_example():
    model_instances, tokenizer = ModelFactory.create_model("llama3_8b", rank=0)
    model = model_instances[0].model

    # Collect special token ids for exclusion
    exclude_token_ids = []
    if tokenizer.pad_token_id is not None:
        exclude_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        exclude_token_ids.append(tokenizer.unk_token_id)
    if tokenizer.eos_token_id is not None:
        exclude_token_ids.append(tokenizer.eos_token_id)

    # Patch Batch class to use these exclude_token_ids
    import src.batching.batch as batch_mod
    batch_mod.Batch.exclude_token_ids = exclude_token_ids

    # Derive leading punctuation (runs of only '!') token ids to avoid at start
    punctuation_ids = []
    try:
        vocab_size = len(tokenizer)
        for tid in range(vocab_size):
            decoded = tokenizer.decode([tid]).strip()
            if decoded and all(ch == '!' for ch in decoded) and 0 < len(decoded) <= 5:
                punctuation_ids.append(tid)
    except Exception:
        pass
    batch_mod.Batch.leading_exclude_token_ids = punctuation_ids

    prompts = [
        "Answer in one word. What is the last word in the sentence that you are reading right now?",
        "Write a short story about a robot learning to paint.",
        "Explain the theory of relativity in simple terms.",
        "List three benefits of regular exercise.",
        "Describe the process of photosynthesis in detail, including the role of chlorophyll and sunlight.",
        "Summarize the plot of 'Pride and Prejudice' in one sentence.",
        "What are the main differences between classical and quantum computing?",
        "How does the human brain process information?",
        "What are the key principles of effective time management?",
        "Explain the significance of the Turing test in artificial intelligence.",
        "Describe the impact of climate change on global ecosystems.",
        "Discuss the ethical implications of genetic engineering in humans, including potential benefits and risks."
    ]
    
    print("🚀 Starting Llama 3 8B Inference")
    print(f"📝 Processing {len(prompts)} requests...")
    
    # Create engine using factory - now using standard model engine 
    engine = EngineFactory.create_engine("model", model=model)
    
    # Create scheduler using factory with engine instead of model
    scheduler = SchedulerFactory.create_scheduler(
        name="fcfs",
        engine=engine,
        tokenizer=tokenizer,
        batch_size=4
    )
    
    for prompt in prompts:
        scheduler.add_sequence_to_queue(prompt)
    
    import time
    start_time = time.time()
    results = scheduler.run_scheduler()
    end_time = time.time()
    
    # Mark completion times for clean output
    for seq in results:
        if not hasattr(seq, 'finish_time') or seq.finish_time is None:
            seq.finish_time = end_time
    
    print(f"\n✅ Completed {len(results)} requests in {end_time - start_time:.2f}s")
    
    # Print clean results
    formatter = CleanOutputFormatter()
    formatter.print_request_summary(results, tokenizer)
    
    # Print individual results
    for seq in results:
        formatter.print_request_result(seq, tokenizer, show_prompt=True)


if __name__ == "__main__":
    usage_example()
