import queue

import torch

from sequence import Sequence

class Scheduler:
    def __init__(self, model, tokenizer, batch_size=9):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_queue = queue.Queue()
        self.num_iterations = 10

    def add_sequence_to_queue(self, prompt, stage="prefill"):
        seq = Sequence(prompt, self.tokenizer, stage)
        self.sequence_queue.put(seq)

         
    def run_scheduler(self):
        dones = []
        while not self.sequence_queue.empty():
            batch = []
            while len(batch) < self.batch_size and not self.sequence_queue.empty():
                batch.append(self.sequence_queue.get())
            
            if len(batch) == 0:
                break
            
            with torch.no_grad():
                for i in range(self.num_iterations):
                    outputs = self.model(sequences=batch, use_cache=True)
                    # if batch[0].stage == "prefill":
                    #     for seq in batch:
                    #         seq.stage = "decode"

                    # logits = outputs.logits
                    # kv_cache = outputs.past_key_values
                    # last_token_logits = logits[:, -1, :]
                    # next_token_ids = torch.argmax(last_token_logits, dim=-1).unsqueeze(-1)

                    # seq.update(next_token_ids, kv_cache)
                dones.extend(outputs)
        return dones