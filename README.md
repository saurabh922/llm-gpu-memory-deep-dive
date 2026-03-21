# llm-gpu-memory-deep-dive
A deep dive into GPU memory usage in transformers, comparing inference and training through theory, exact parameter math, and PyTorch experiments.


---

## Inference Experiment

### Objective

Understand how GPU memory is consumed during inference.

https://medium.com/@sabu.for.ai/understanding-llm-gpu-inference-vram-kv-cache-and-vllm-explained-with-mistral-7b-ea73c562f312

### Observations

- Memory is dominated by model weights and KV cache
- KV cache grows with sequence length
- No gradients or optimizer states are involved

### Key Insight

Inference memory remains relatively stable compared to training and is primarily influenced by sequence length and caching mechanisms.

---

## Training Experiment

https://medium.com/@sabu.for.ai/understanding-llm-gpu-training-training-transformers-needs-6-8-more-gpu-memory-than-inference-b17aafb1f7e3

### Model Configuration

A small transformer model was used for controlled experimentation:

- d_model = 512
- num_heads = 4
- encoder_layers = 4
- decoder_layers = 4
- vocab_size = 10,000
- total parameters ≈ 39.6M

### Weight Memory (FP32)

PyTorch uses float32 by default:

```
39.6M × 4 bytes ≈ 158 MB (~160 MB)
```

---

## Memory Measurement

GPU memory was tracked using:

```python
torch.cuda.memory_allocated()
torch.cuda.max_memory_allocated()