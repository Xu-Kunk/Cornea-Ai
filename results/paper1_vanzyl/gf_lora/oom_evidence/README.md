# Geneformer LoRA — OOM evidence

## Purpose

This directory records the evidence that the Geneformer LoRA arm could not be
evaluated because of GPU memory constraints (Limitations, fourth point). The
identity check passed, but the first forward pass raised an out-of-memory error.

## Run configuration

- Device: `cuda`, seed: 42
- `n_author_classes = 30`
- `n_total_cells = 80817`
- Locked validation split: 76,570 train / 4,247 val

## LoRA configuration

- 12 layers wrapped, `r = 8`, `alpha = 16`
- Trainable parameters: head 0.023M + adapters 0.442M

## Identity check (pre-gate)

- With B = 0, maximum absolute difference between LoRA output and base output:
  **6.45e-06** → **PASS**
  (Proves the wrapper does not alter baseline behaviour.)

## Failure point

- The first forward pass raised `torch.OutOfMemoryError`.
- CUDA allocator: `expandable_segments: memory mapping failed with OOM on
  device 0 while trying to map 20971520 bytes (free: 19595264, total:
  16611278848)`.
- Total device memory ≈ 16.6 GB.

## Conclusion

On this 16 GB device, Geneformer LoRA could not be evaluated. The
cross-architecture LoRA claim therefore rests on scGPT-human and CellPLM.
