from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

def llama8b_tensor_parallel_plan(model):
    plan = {}
    for idx, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{idx}"
        plan[f"{prefix}.self_attn.q_proj"] = ColwiseParallel()
        plan[f"{prefix}.self_attn.k_proj"] = ColwiseParallel()
        plan[f"{prefix}.self_attn.v_proj"] = ColwiseParallel()
        plan[f"{prefix}.self_attn.o_proj"] = RowwiseParallel()
        plan[f"{prefix}.mlp.gate_proj"] = ColwiseParallel()
        plan[f"{prefix}.mlp.down_proj"] = RowwiseParallel()
        plan[f"{prefix}.mlp.up_proj"] = ColwiseParallel()
    return plan
