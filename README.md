# 🤖 Sarcasm & Sentiment Classification using Transformers

This project demonstrates how to fine-tune Hugging Face Transformers (`BERT` and `DistilBERT`) for **sarcasm detection** and **sentiment analysis** on custom datasets using **GPU acceleration**.

> 🚀 Built with 🤗 Transformers, PyTorch, and Hugging Face Datasets

---

## 📌 Features

- ✅ Fine-tunes on both **IMDB sentiment** and **Sarcasm Headlines** datasets
- ✅ Fully GPU-accelerated training with automatic CUDA detection
- ✅ Balanced sarcasm dataset for improved performance
- ✅ Human-readable labels (e.g., "Positive", "Sarcastic")
- ✅ Supports sarcasm-aware post-processing for edge cases
- ✅ Saves and reuses fine-tuned models locally
- ✅ Logs accuracy and F1 score during evaluation
- ✅ Clean and modular code with reproducible setup

---

## 📂 Dataset Sources

- **IMDB Sentiment Dataset** – from Hugging Face Hub (`imdb`)
- **Sarcasm Headlines Dataset** – local JSON:  
  [`Sarcasm_Headlines_Dataset_v2.json`](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)

---

## 🔧 Setup Instructions

```bash
# Create and activate virtual environment
conda create -n sarcasm_transformers python=3.10 -y
conda activate sarcasm_transformers

# Install core dependencies
pip install transformers[torch] datasets scikit-learn accelerate

# Optional: Download CUDA-enabled PyTorch if not already installed
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

