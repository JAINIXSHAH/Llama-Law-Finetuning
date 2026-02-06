# Fine-Tuning Llama 2 on Indian Legal Dataset

This notebook demonstrates how to fine-tune Llama 2 (7B chat model) on Indian legal documents and case law. The goal is to create a model that can answer questions about Indian law based on legal contexts and precedents.

## What's This Project About?

We're training a legal assistant that understands Indian law. The model learns from question-answer pairs that include legal context (like excerpts from court judgments) and needs to provide accurate answers based on that context. Think of it as teaching the model to read legal documents and answer questions like a law student would.

## Requirements

You'll need:
- A GPU (works on Colab's free tier with T4)
- Hugging Face account with access token
- Weights & Biases account (optional, for training metrics)
- The notebook uses 1,000 samples from the training dataset to keep things manageable

## Installation

Start by installing Unsloth, which optimizes LLM training for better speed and memory efficiency:

```bash
pip install unsloth unsloth_zoo --no-cache-dir
```

## The Dataset

We're using `Prarabdha/indian-legal-supervised-fine-tuning-data`, which contains Indian legal Q&A pairs. Each example has:
- **context**: Excerpt from legal documents, judgments, or case law
- **question**: A specific question about the legal context
- **response**: The correct answer based on the context

The dataset is loaded in streaming mode and we take the first 1,000 examples for training (you can adjust this number based on your needs and compute budget).

## How It Works

### 1. Data Preparation

The training data is formatted using the Llama 2 chat template:

```
<s>[INST] <<SYS>>
You are a helpful assistant specialized in Indian law.
<</SYS>>

Context:
{legal_context}

Question:
{question}
[/INST]
{answer}</s>
```

This format teaches the model to:
- Understand its role as a legal assistant
- Read and comprehend the legal context
- Answer questions accurately based on that context

### 2. Loading the Base Model

We load Llama 2 7B Chat with these settings:
- 4-bit quantization (saves memory, makes it runnable on free Colab)
- Max sequence length: 2048 tokens
- Gradient checkpointing enabled
- Random seed: 3407 (for reproducibility)

### 3. LoRA Configuration

Instead of fine-tuning all 7 billion parameters, we use LoRA (Low-Rank Adaptation) which is much more efficient:

- Rank (r): 16
- Alpha: 16
- Target modules: All attention and MLP layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Dropout: 0 (no dropout)
- Gradient checkpointing: Enabled via Unsloth

This approach trains only a small fraction of the parameters while maintaining good performance.

### 4. Training Setup

The training configuration:
- Batch size: 1 per device
- Gradient accumulation: 4 steps (effective batch size of 4)
- Learning rate: 2e-4
- Epochs: 1 (one pass through the data)
- No sequence packing (each example processed separately)

Training progress is logged to Weights & Biases if configured.

### 5. Testing the Model

The notebook includes before/after comparisons. There's a test example about government measures regarding construction materials at a site (appears to be related to the Ram Janmabhoomi case). You can see how the model's responses improve after fine-tuning.

## Project Structure

The workflow is pretty straightforward:
1. Load and preprocess the legal dataset
2. Format data using Llama 2's chat template
3. Load the base model with 4-bit quantization
4. Apply LoRA adapters
5. Train using supervised fine-tuning
6. Test on example legal questions

## Key Features

**Streaming Dataset Loading**: The dataset is loaded in streaming mode, which means you can work with large datasets without loading everything into memory at once.

**Efficient Training**: Using 4-bit quantization + LoRA means you can fine-tune a 7B model on free Colab GPUs.

**Domain-Specific**: The model is specifically trained on Indian legal content, making it more reliable for questions about Indian law compared to general-purpose models.

## What Can You Do With This?

After training, you get a model that can:
- Read excerpts from legal documents
- Answer questions about Indian law
- Provide context-based legal reasoning
- Handle questions about specific cases and judgments

## Practical Notes

- The notebook uses 1,000 samples for training - you can increase this by changing `islice(dataset_stream, 1000)` to a higher number
- If you hit memory issues, try reducing the max sequence length from 2048 to something smaller like 1024
- The model learns to stick to the provided context, which is important for legal applications (you don't want it hallucinating legal facts)
- Training takes roughly 30-60 minutes on a T4 GPU for 1,000 samples

## Before vs After Training

The notebook includes comparison tests showing how the model's responses improve. Before training, the base Llama 2 model might give generic or incorrect answers. After fine-tuning on legal data, it provides more accurate, context-based responses that align with Indian legal reasoning.

## Why This Matters

Legal text is complex and domain-specific. General-purpose language models often struggle with legal queries because:
- Legal language is highly specialized
- Context matters immensely (what one judgment says might differ from another)
- Accuracy is critical (you can't afford hallucinations in legal applications)

Fine-tuning on Indian legal data helps the model understand these nuances and provide more reliable answers.

## Dataset Citation

This project uses the Indian Legal Supervised Fine-Tuning dataset by Prarabdha, which contains high-quality legal Q&A pairs based on Indian law and court judgments.
