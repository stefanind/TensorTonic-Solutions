# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Stefan Indic's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/stefanindic.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Evaluate a Scalar Neuron | Evaluate a scalar PyTorch tanh neuron from aligned inputs, weights, and bias using promoted floating-point types. | https://www.tensortonic.com/problems/autograd-l04-neuron-forward |
| Differentiate a Tanh Activation | Evaluate scalar tanh and manually combine its local derivative with an upstream gradient. | https://www.tensortonic.com/problems/autograd-l04-tanh-forward-backward |
| Named-Dimension Batched Attention Scores | Compute batched multi-head query-key scores by contracting only the head-width dimension. | https://www.tensortonic.com/problems/cs336-l02-einsum-attention-scores |
| Gradient Accumulation Equivalence | Combine mean-loss gradients from unequal microbatches into one full-batch mean gradient, then apply a single SGD update. | https://www.tensortonic.com/problems/cs336-l02-gradient-accumulation-step |
| RMSNorm Forward Pass | Normalize each final-dimension vector by its root mean square and apply the learned scale without mean subtraction. | https://www.tensortonic.com/problems/cs336-l03-rmsnorm-forward |
| Rotary Query and Key Embeddings | Rotate each adjacent coordinate pair of query and key vectors by a position-dependent angle. | https://www.tensortonic.com/problems/cs336-l03-rotary-query-key-embeddings |
| Masked Triton GELU Kernel | Implement masked tanh-approximate GELU in Triton for contiguous CUDA tensors, partial final blocks, and multiple dtypes. | https://www.tensortonic.com/problems/cs336-l06-triton-masked-gelu |
| Triton Row-Wise Softmax | Implement stable row-wise softmax in Triton for padded CUDA rows, partial tiles, and float32, float16, or bfloat16 data. | https://www.tensortonic.com/problems/cs336-l06-triton-row-softmax |
| Tiled Triton Row Sum | Implement tiled row reduction in Triton for strided CUDA tensors, fixed-width tiles, and float32 output sums. | https://www.tensortonic.com/problems/cs336-l06-triton-tiled-row-sum |
| Calculate Collective Effective Bandwidth | Calculate ring-equivalent algorithm bytes per rank and effective bandwidth for a supported collective. | https://www.tensortonic.com/problems/cs336-l07-collective-bandwidth |
| Account for ZeRO Memory by Stage | Calculate persistent model-state bytes per rank for DDP and ZeRO stages 1, 2, and 3 with ceiling division for uneven shards. | https://www.tensortonic.com/problems/cs336-l08-zero-memory-accounting |
| Estimate Critical Batch Size | Convert a batch-size sweep into examples consumed at a target loss and estimate the critical batch size from the minimum reached step and example counts. | https://www.tensortonic.com/problems/cs336-l09-critical-batch-size |
| Orthogonalize a Muon Momentum Update | Form matrix momentum, replace its compact singular values with one, and apply the resulting spectral update. | https://www.tensortonic.com/problems/cs336-l11-muon-spectral-update |
| Build an Assistant-Only SFT Loss Mask | Build a shifted supervised-fine-tuning loss mask that trains only on assistant tokens and ignores padding and controls. | https://www.tensortonic.com/problems/cs336-l15-assistant-only-sft-mask |
| Patchify a Batch of Images | Split PyTorch image batches into a row-major grid of non-overlapping flattened Vision Transformer patches. | https://www.tensortonic.com/problems/cs336-l17-vision-transformer-patchify |
| Mean IoU for Segmentation | Compute the mean Intersection-over-Union (mIoU) for a semantic segmentation prediction. | https://www.tensortonic.com/problems/cv-miou |
| ViT Patchify | Split image batches into row-major non-overlapping patches and flatten each patch in spatial-then-channel order. | https://www.tensortonic.com/problems/cv-patchify |
| Top-K Classification Accuracy | Given logits of shape (N, K) and integer targets of shape (N,) with each target in [0, K), compute the top-k classification accuracy. | https://www.tensortonic.com/problems/cv-topk-accuracy |
| Activation Functions | Implement ReLU, sigmoid, tanh, Leaky ReLU, GELU, and Swish with their analytical derivatives. | https://www.tensortonic.com/problems/dl-activation-functions |
| Computational Graph & Autograd | Build a minimal autograd engine that performs forward and backward passes on a computational graph. | https://www.tensortonic.com/problems/dl-autograd |
| Backpropagation (Single Hidden Layer) | Backpropagate MSE loss through a one-hidden-layer NumPy network to compute gradients for every weight and bias. | https://www.tensortonic.com/problems/dl-backpropagation |
| Batch Normalization | Implement batch normalization for training and inference, including batch statistics and running-statistic updates. | https://www.tensortonic.com/problems/dl-batch-normalization |
| 2D Convolution from Scratch | Implement multichannel two-dimensional cross-correlation from scratch with multiple filters, stride, padding, and bias. | https://www.tensortonic.com/problems/dl-convolution-operation |
| Dropout | Apply inverted dropout from a supplied binary mask during training and preserve the input unchanged during evaluation. | https://www.tensortonic.com/problems/dl-dropout |
| Multi-Layer Perceptron (Forward Pass) | Implement the forward pass of a multi-layer perceptron (MLP) with arbitrary depth and width. | https://www.tensortonic.com/problems/dl-forward-pass |
| Loss Functions | Implement MSE, binary cross-entropy, categorical cross-entropy, and Huber losses from supplied predictions and targets. | https://www.tensortonic.com/problems/dl-loss-functions |
| Mini-Batch Training Loop | Train a NumPy multilayer perceptron over ordered mini-batches with forward passes, backpropagation, and SGD updates. | https://www.tensortonic.com/problems/dl-mini-batch-training |
| Perceptron | Train a binary perceptron from zero-initialized weights using ordered samples, step predictions, and error-correction updates. | https://www.tensortonic.com/problems/dl-perceptron |
| Pooling Layers | Implement channelwise two-dimensional max and average pooling with configurable kernel and stride. | https://www.tensortonic.com/problems/dl-pooling-layers |
| Weight Initialization | Compute per-layer NumPy weight initialization parameters from network dimensions and the selected initialization method. | https://www.tensortonic.com/problems/dl-weight-initialization |
| Implement Autoregressive Decoding with a KV Cache | Decode autoregressively with an append-only key-value cache so previously processed tokens are never recomputed. | https://www.tensortonic.com/problems/inference-autoregressive-kv-cache |
| Implement Block-Wise FP8 Quantization | Compute one scale per block from that block's own absolute maximum, and scale each block into the [-fp8 max, fp8 max] range with saturation. | https://www.tensortonic.com/problems/inference-blockwise-fp8-quantization |
| Simulate Chunked Prefill Scheduling | Simulate chunked prefill scheduling with strict decode priority, bounded token budgets, and request arrival order. | https://www.tensortonic.com/problems/inference-chunked-prefill-scheduling |
| Simulate Continuous Batching | Simulate continuous batching by admitting requests in arrival order and replacing completed sequences during decoding. | https://www.tensortonic.com/problems/inference-continuous-batching |
| Simulate Disaggregated Prefill and Decode Serving | Simulate Disaggregated Prefill and Decode Serving, and return selected prefill and decode replica indices, TTFT, and completion time for every request. | https://www.tensortonic.com/problems/inference-disaggregated-prefill-decode |
| Implement Dynamic Request Batching | Simulate a dynamic batching scheduler: requests arrive over time and are grouped into a forming batch. | https://www.tensortonic.com/problems/inference-dynamic-batching |
| Implement Expert Parallel Routing Across GPUs | Route token-expert assignments to destination GPU ranks with stable ordering, send counts, and an inverse permutation. | https://www.tensortonic.com/problems/inference-expert-parallel-routing |
| Implement FlashAttention with Online Softmax | Implement tiled attention with online softmax statistics that agrees numerically with dense scaled attention. | https://www.tensortonic.com/problems/inference-flash-attention-online-softmax |
| Implement Grouped-Query Attention (GQA) | Implement grouped-query attention by mapping query heads onto fewer key-value heads with validated head divisibility. | https://www.tensortonic.com/problems/inference-grouped-query-attention |
| Implement Group-Wise INT4 Quantization | Quantize two-dimensional weights in contiguous groups with signed INT4 values and one scale per group. | https://www.tensortonic.com/problems/inference-groupwise-int4-quantization |
| Calculate KV Cache Memory for MHA, MQA, GQA, and MLA | Compute the total KV-cache memory, in bytes, for a full sequence under four attention variants: MHA, MQA, GQA, and MLA. | https://www.tensortonic.com/problems/inference-kv-cache-memory |
| Calculate P50, P95, and P99 Inference Latency | Compute the 50th, 95th, and 99th percentile of a set of observed inference latencies, using linear interpolation between the two closest ranks. | https://www.tensortonic.com/problems/inference-latency-percentiles |
| Estimate Model Memory and Minimum GPU Count | Estimate model-serving memory from weights, key-value cache, activations, overhead, and per-GPU usable capacity. | https://www.tensortonic.com/problems/inference-model-memory-gpu-count |
| Implement MoE Token Dispatch and Expert Aggregation | Run each expert's feed-forward computation only on the tokens routed to it, and scatter-add the routing-weighted results back into token order. | https://www.tensortonic.com/problems/inference-moe-dispatch-aggregation |
| Implement Sparse MoE Top-k Expert Routing | Select the top k highest-scoring experts per token and compute routing weights as a softmax over only those selected logits. | https://www.tensortonic.com/problems/inference-moe-top-k-routing |
| Implement Multi-Head Attention (MHA) | Split into h heads, run scaled dot-product attention per head with an optional causal mask, concatenate the heads, and apply an output projection. | https://www.tensortonic.com/problems/inference-multi-head-attention |
| Implement Multi-Head Latent Attention (MLA) | Implement Multi-Head Latent Attention (MLA), and return both the projected attention output and the compressed latent tensor. | https://www.tensortonic.com/problems/inference-multi-head-latent-attention |
| Implement Multi-Query Attention (MQA) | Implement multi-query attention with separate query heads, shared key-value heads, optional masking, and output projection. | https://www.tensortonic.com/problems/inference-multi-query-attention |
| Implement PagedAttention Block Allocation | Assign each sequence enough fixed-size blocks to hold its tokens, drawn from the free pool in order. | https://www.tensortonic.com/problems/inference-paged-attention-allocation |
| Implement Per-Channel Weight Quantization | Quantize weights independently along a selected channel axis, then dequantize with channel-specific scales. | https://www.tensortonic.com/problems/inference-per-channel-quantization |
| Analyze Prefill and Decode with the Roofline Model | Analyze the prefill and decode phases of transformer inference using the roofline performance model. | https://www.tensortonic.com/problems/inference-prefill-decode-roofline |
| Implement Prefix Cache Matching and Reuse | Find the longest exactly-matching prefix, measured in whole blocks, among all candidates. | https://www.tensortonic.com/problems/inference-prefix-cache-reuse |
| Implement Rotary Position Embeddings for Decoding | Apply rotary position embeddings to query and key tensors at supplied absolute decoding positions. | https://www.tensortonic.com/problems/inference-rotary-position-embeddings |
| Implement Scaled Dot-Product Attention | Implement batched scaled dot-product attention for self- and cross-attention with optional masks and stable softmax. | https://www.tensortonic.com/problems/inference-scaled-dot-product-attention |
| Compute TTFT, TPOT, ITL, and Token Throughput | Compute TTFT, TPOT, inter-token latency, and token throughput from request arrival and token timestamp data. | https://www.tensortonic.com/problems/inference-serving-metrics |
| Implement Speculative Decoding and Token Verification | Verify speculative draft tokens against target probabilities and sample the corrected or bonus continuation exactly. | https://www.tensortonic.com/problems/inference-speculative-decoding-verification |
| Implement Symmetric INT8 Quantization | Quantize finite tensors to symmetric INT8 values with an absolute-maximum scale and reconstruct dequantized outputs. | https://www.tensortonic.com/problems/inference-symmetric-int8-quantization |
| Calculate Tensor Parallel Communication Cost | Calculate tensor-parallel collective communication volume and latency from message size, world size, bandwidth, and calls. | https://www.tensortonic.com/problems/inference-tensor-parallel-communication |
| Optimize Throughput and Cost Under a Latency SLO | Select the cheapest configuration that meets both a required throughput and a maximum P95 latency SLO. | https://www.tensortonic.com/problems/inference-throughput-cost-slo |
| Implement Temperature, Top-k, and Top-p Sampling | Sample tokens from batched logits using temperature scaling, top-k filtering, top-p filtering, and deterministic randomness. | https://www.tensortonic.com/problems/inference-token-sampling |
| Implement Dot Product | Compute the algebraic dot product and geometric angle relationship for two equal-length NumPy vectors. | https://www.tensortonic.com/problems/la-dot-product |
| Matrix Transpose | Transpose a rectangular NumPy matrix by swapping its row and column axes without changing element values. | https://www.tensortonic.com/problems/la-matrix-transpose |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Activation Functions | Implement four common activation functions from scratch using basic PyTorch tensor operations (no torch.nn module). | https://www.tensortonic.com/problems/pytorch-activation-function-from-scratch |
| Attention Mechanism from Scratch | Implement the scaled dot-product attention mechanism, a core building block of the Transformer architecture. | https://www.tensortonic.com/problems/pytorch-attention-from-scratch |
| Balanced DataLoader | Build a PyTorch DataLoader that balances class sampling with per-example weights derived from label frequencies. | https://www.tensortonic.com/problems/pytorch-balanced-dataloader |
| Basic Autograd | Use PyTorch autograd to evaluate a scalar function and return its derivative at every supplied input value. | https://www.tensortonic.com/problems/pytorch-basic-autograd |
| Batch Normalization | Normalize each feature across the batch, then scale and shift using learnable parameters. | https://www.tensortonic.com/problems/pytorch-batch-normalization |
| Beam Search Decoding | Beam search is a heuristic search algorithm used in sequence generation tasks such as machine translation, text generation, and speech recognition. | https://www.tensortonic.com/problems/pytorch-beam-search |
| Simple Neural Network | Implement a class SimpleNet subclassing nn.Module with two linear layers and ReLU between them. | https://www.tensortonic.com/problems/pytorch-build-simple-nn-from-scratch |
| Conv2d from Scratch | Implement a PyTorch Conv2d module from tensor operations with configurable channels, kernel, stride, padding, and bias. | https://www.tensortonic.com/problems/pytorch-conv2d-from-scratch |
| Custom Dataset Class | Implement a PyTorch Dataset over row records with indexed feature tensors and labels. | https://www.tensortonic.com/problems/pytorch-custom-dataclass |
| Custom Linear Layer | Implement a custom linear layer that computes the affine transformation without using any built-in linear layer. | https://www.tensortonic.com/problems/pytorch-custom-linear-layer |
| Custom SGD with Momentum | Implement momentum SGD by subclassing the PyTorch optimizer interface and maintaining per-parameter velocity. | https://www.tensortonic.com/problems/pytorch-custom-optimizer |
| Dropout from Scratch | Implement PyTorch inverted dropout from a supplied mask during training while returning inputs unchanged in evaluation mode. | https://www.tensortonic.com/problems/pytorch-dropout-from-scratch |
| Early Stopping | Train a PyTorch model with validation monitoring and stop after the configured number of unimproved epochs. | https://www.tensortonic.com/problems/pytorch-early-stopping |
| Gradient Accumulation | Simulate gradient accumulation over multiple micro-batches, and return the final weights and last averaged gradient. | https://www.tensortonic.com/problems/pytorch-gradient-accumulation |
| Loss Functions | Implement three common loss functions from scratch using PyTorch tensor operations: mean squared error, cross-entropy, and Huber loss. | https://www.tensortonic.com/problems/pytorch-loss-functions |
| Learning Rate Warmup Scheduler | Implement a function that computes a learning rate schedule combining linear warmup with cosine decay. | https://www.tensortonic.com/problems/pytorch-lr-warmup-scheduler |
| LSTM Cell from Scratch | Implement a single LSTM (Long Short-Term Memory) cell that processes one time step of input. | https://www.tensortonic.com/problems/pytorch-lstm-cell-from-scratch |
| Manual Weight Update | Perform a PyTorch training step with manual parameter updates after backpropagation, without an optimizer object. | https://www.tensortonic.com/problems/pytorch-manual-weight-update |
| Masked Causal Attention | Implement scaled dot-product attention with a causal mask that prevents each position from attending to future positions. | https://www.tensortonic.com/problems/pytorch-masked-causal-attention |
| Mini Training Loop | Run one complete PyTorch training epoch over a DataLoader and return the mean batch loss. | https://www.tensortonic.com/problems/pytorch-mini-training |
| Multi-Head Attention | Implement PyTorch multi-head attention with head splitting, scaled softmax attention, concatenation, and output projection. | https://www.tensortonic.com/problems/pytorch-multi-head-attention |
| Optimizer Scheduler | Train with a PyTorch optimizer and StepLR schedule, recording the learning rate applied at each epoch. | https://www.tensortonic.com/problems/pytorch-optimizer-scheduler |
| Residual Block | Implement a PyTorch residual block with two padded convolutions, batch normalization, ReLU, and an identity shortcut. | https://www.tensortonic.com/problems/pytorch-residual-block |
| RNN Cell from Scratch | Implement a vanilla PyTorch RNN cell that combines current inputs and previous hidden states with a tanh update. | https://www.tensortonic.com/problems/pytorch-rnn-cell-from-scratch |
| Softmax from Scratch | Implement numerically stable batched softmax in PyTorch by shifting logits before exponentiation and normalization. | https://www.tensortonic.com/problems/pytorch-softmax-from-scratch |
| Tensor Operations | Perform common element-wise and matrix tensor operations: add, multiply, matmul, power, and max. | https://www.tensortonic.com/problems/pytorch-tensor-arithmetic |
| Tensor Factory | Create PyTorch tensors with zeros, ones, or a constant fill value using the requested shape and dtype. | https://www.tensortonic.com/problems/pytorch-tensor-creation |
| Tensor Shape Manipulation | Reshape tensors using three common PyTorch operations: flatten to collapse into 1D, squeeze to remove size-1 dimensions. | https://www.tensortonic.com/problems/pytorch-tensor-reshape |
| Transform Pipeline | Implement a callable class that converts a raw image tensor into a normalized, channel-first tensor ready for a neural network. | https://www.tensortonic.com/problems/pytorch-transforms-pipeline |
| Weight Initialization | Implement a function that initializes a weight tensor using one of four standard initialization methods. | https://www.tensortonic.com/problems/pytorch-weight-initialization |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Block-Pointer Matmul | Implement tiled Triton matrix multiplication with block pointers, boundary checks, and tail-safe loads and stores. | https://www.tensortonic.com/problems/triton-block-pointer-matmul |
| Dropout (Inverted Scaling) | Implement inverted dropout in Triton with a supplied mask, register scaling, and tail-safe tiled memory access. | https://www.tensortonic.com/problems/triton-dropout |
| Fused Multiply-Add | Implement a Triton fused multiply-add kernel with contiguous tiles, hardware FMA, and masked tail handling. | https://www.tensortonic.com/problems/triton-fused-multiply-add |
| Fused Row-Wise Softmax | Implement fused row-wise softmax in Triton with stable max subtraction, register reductions, and masked column tails. | https://www.tensortonic.com/problems/triton-fused-softmax |
| GELU | Implement exact GELU activation in Triton with device error-function math and masked contiguous tiles. | https://www.tensortonic.com/problems/triton-gelu |
| GEMV: Matrix Vector Product | Implement Triton matrix-vector multiplication with row-block programs, float32 accumulation, and masked matrix tails. | https://www.tensortonic.com/problems/triton-gemv |
| KV Cache Append | Append one autoregressive decoding row to key and value caches in Triton without modifying other cache positions. | https://www.tensortonic.com/problems/triton-kv-append |
| L2 Vector Norm | Compute a Triton L2 vector norm with tiled sum-of-squares reduction, atomic accumulation, and masked tail lanes. | https://www.tensortonic.com/problems/triton-l2-norm |
| LayerNorm Forward | Implement LayerNorm forward in Triton with per-row mean and variance reductions, affine parameters, and masked tails. | https://www.tensortonic.com/problems/triton-layernorm |
| Row-Wise LogSumExp | Implement numerically stable row-wise LogSumExp in Triton with max subtraction and masked register reductions. | https://www.tensortonic.com/problems/triton-logsumexp |
| Autotuned Matrix Multiplication | Autotune Triton matrix multiplication across tile and pipeline configurations while preserving masked boundary handling. | https://www.tensortonic.com/problems/triton-matmul-autotune |
| Vector Max Reduction | Compute a vector maximum with one Triton reduction program and masked tail lanes that cannot win comparisons. | https://www.tensortonic.com/problems/triton-max |
| Single-Pass Mean and Variance | Compute population mean and variance in Triton with single-pass statistics, atomic partials, and masked tails. | https://www.tensortonic.com/problems/triton-mean-variance |
| Online Softmax | Implement chunked online softmax in Triton with running maxima and denominators followed by a normalized output pass. | https://www.tensortonic.com/problems/triton-online-softmax |
| ReLU | Implement ReLU activation in Triton with contiguous program tiles, branch-free rectification, and masked tails. | https://www.tensortonic.com/problems/triton-relu |
| RMSNorm Forward | Implement RMSNorm forward in Triton with per-row square reduction, numerical stability, scaling, and masked tails. | https://www.tensortonic.com/problems/triton-rmsnorm |
| Rotary Position Embedding | Implement Rotary Position Embeddings in Triton with per-token pair rotations, precomputed sine and cosine, and tail masks. | https://www.tensortonic.com/problems/triton-rope |
| Scaled Dot-Product Attention | Implement three-stage scaled dot-product attention in Triton with tiled scores, stable softmax, and value aggregation. | https://www.tensortonic.com/problems/triton-scaled-dot-product-attention |
| SiLU | Implement fused SiLU or Swish activation in Triton with contiguous tiles, sigmoid weighting, and masked tails. | https://www.tensortonic.com/problems/triton-silu |
| Vector Sum Reduction | Implement tiled vector sum reduction in Triton with register partials, atomic accumulation, and masked tail lanes. | https://www.tensortonic.com/problems/triton-sum |
| Tiled Transpose | Implement tiled matrix transpose in Triton by swapping load and store strides with masked boundary tiles. | https://www.tensortonic.com/problems/triton-transpose |
| Vector Addition | Implement elementwise vector addition in Triton with contiguous program tiles and safe masking for partial tails. | https://www.tensortonic.com/problems/triton-vector-addition |
| Vectorized Vector Add | Implement vector addition in Triton with larger per-program tiles to reduce launch overhead while masking the final tail. | https://www.tensortonic.com/problems/triton-vectorized-load |
| Forward Diffusion Process | Implement the DDPM forward diffusion process by mixing clean samples with Gaussian noise at a selected timestep. | https://www.tensortonic.com/research/ddpm/ddpm-forward |
| DDPM Training Loss | Compute the DDPM training objective as mean squared error between sampled noise and the model's noise prediction. | https://www.tensortonic.com/research/ddpm/ddpm-loss |
| Reverse Diffusion Process | Implement one DDPM reverse-process step using the model's predicted noise and the schedule coefficients. | https://www.tensortonic.com/research/ddpm/ddpm-reverse |
| DDPM Sampling | Implement iterative DDPM sampling from Gaussian noise by applying the learned reverse process over all timesteps. | https://www.tensortonic.com/research/ddpm/ddpm-sampling |
| Noise Schedule | Generate a DDPM noise schedule with beta values and cumulative alpha products used across diffusion timesteps. | https://www.tensortonic.com/research/ddpm/ddpm-schedule |
| Full Attention Residuals | Implement Kimi K3 full attention residuals by mixing normalized layer outputs with learned residual coefficients. | https://www.tensortonic.com/research/kimik3/k3-full-attnres |
| Gated MLA | Build Kimi K3 Gated MLA with low-rank query and KV projections, per-head gating, RoPE inputs, and latent caching. | https://www.tensortonic.com/research/kimik3/k3-gated-mla |
| KDA Recurrence | Implement Kimi K3's KDA recurrence with gated state updates, normalized queries and keys, and ordered token readouts. | https://www.tensortonic.com/research/kimik3/k3-kda-recurrence |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/stefanindic)
<!-- tensortonic:end -->
