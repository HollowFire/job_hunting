# LLM inference optimization

https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

## Attention optimization

### Multi-head attention

### Multi-query attention

### Grouped-query attention

### Flash Attention

- mathematically identical to standard multi-head attention

### Paged Attention

- With sequence length unpredictable, a reservation of 2048 is made in memory.
- Enables KV to be stored in non-contiguous memory.


##  Model optimization

### Quantization

Benifits include:
- less memory usage
- more parameters
- more effcient parameter transfer (alleviate bandwith limitation)

methods include:
- parameter quantization
- activation quantization: activation often contains outliers, larger dynamic range

### Sparsity

certain hardware has acceleration for structured sparsity

### Distillation


## Model Serving

### In-flight batching
Rather than waiting for the whole batch to finish before moving on to the next set of requests, the server runtime immediately evicts finished sequences from the batch. It then begins executing new requests while other requests are still in flight

### Sepculative inference
