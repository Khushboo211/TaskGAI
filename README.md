# LSTM Text Generator (Character-Level)

## Overview
This project implements a **memory-safe, character-level LSTM text generator**
using **TensorFlow/Keras**.  
The model is trained on a large corpus of text files and generates new text
using **temperature-based sampling**.

## Features
- Character-level tokenization (no OOV issues)
- Memory-safe data generator
- GPU-ready (CPU-safe fallback)
- Temperature-controlled text generation
- Interview & production-friendly structure

## Dataset
Place `.txt` files inside the `data/` directory.

Example datasets:
- Shakespeare (Project Gutenberg)
- Any public domain text corpus

## Installation
```bash
pip install -r requirements.txt
