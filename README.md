# Prompt-Distillation via Knowledge Distillation

This repository is for our final project MIT 6.4610 Natural Language Processing, where we distill prompt templates and instructions through the framework of knowledge distillation.

## Overview

This project evaluates how well student models (3B parameters) can follow instructions and templates learned from larger teacher models (7-72B parameters) on domains including trivia, math, and general knowledge.

## Components
- **`dataset_generate.ipynb`** - Generate training datasets using teacher models via OpenRouter
- **`student_generate.py`** - Generate responses from student models with/without templates
- **`finetuning.ipynb`** - Fine-tune student models on distilled prompts
- **`llm_judge.py`** - Evaluate model responses using LLM judge (GPT-4o-mini) for correctness and instruction following
- **`experiments/`** - Domain-specific experiments (trivia, math, general) with questions, templates, and model responses

## Authors
Jer Ren Wong
Marin Hristov
Enerelt Delgerdalai
