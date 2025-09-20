import torch
import sys
import gc

sys.path.append("../../")

print("mps" if torch.mps.is_available() else "cpu")


# import torch
from thop import profile
from utils.llms import GPTModel

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

device = torch.device("mps" if torch.mps.is_available() else "cpu")
batch_size = 2
input_tensor = torch.randint(0, 50257, (batch_size, 1024)).to(device)

def memory_gb(size):
    #size in bytes
    s = size / 1024 / 1024 / 1024
    return s

from utils.text import colorize


for size in model_configs:
        BASE_CONFIG.update(model_configs[size])

        model = GPTModel(BASE_CONFIG).bfloat16()
        model.to(device)

        # MACS = multiply-accumulate operations
        # MACS are typically counted as two FLOPS (one multiply and one accumulate)
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops = 2*macs
        print(f"{size:18}: {flops:.1e} FLOPS")

        del model
        
        colorize("current mem usage:",memory_gb(torch.mps.current_allocated_memory()))
        torch.mps.empty_cache()
        colorize("after clearing cache mem usage:", memory_gb(torch.mps.current_allocated_memory()))
        gc.collect()

x = input("exit (:q):")
while x !=":q":
      x = input("exit(:q):")
      