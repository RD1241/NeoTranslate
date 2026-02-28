# ⚡ NeoTranslate  
### Multilingual Neural Machine Translation System (From Scratch)

NeuroLingo-AI is a fully custom-built Neural Machine Translation (NMT) system developed from scratch using TensorFlow, Seq2Seq LSTM with Attention, and autoregressive decoding.

It supports bidirectional translation between:

- 🇬🇧 English  
- 🇮🇳 Hindi  
- 🇮🇳 Punjabi  

This project does **NOT use any pretrained APIs, pretrained embeddings, or external translation services**. The entire dataset, model architecture, and inference pipeline were engineered manually.

---

## 🚀 Features

- 🔁 Bidirectional translation (6 language directions)
- 🧠 LSTM Encoder-Decoder with Attention mechanism
- 📚 Custom dataset generation (240,000+ training pairs)
- 🎯 Autoregressive decoding
- 🌐 FastAPI backend
- 🌌 Futuristic cyberpunk animated UI
- 🔄 Language swap feature
- 🌙 Dark/Light theme toggle
- ⚡ Real-time translation without page reload

---

## 🏗 Architecture

- **Model Type:** Seq2Seq
- **Encoder:** LSTM (256 units)
- **Decoder:** LSTM with Attention
- **Embedding Size:** 128
- **Vocabulary:** ~400+ tokens (expanded grammar dataset)
- **Training Samples:** 240,000
- **Loss:** Sparse Categorical Crossentropy
- **Framework:** TensorFlow 2.15

---

## 📊 Training Performance

| Epoch | Validation Accuracy |
|-------|---------------------|
| 1     | 47% |
| 3     | 72% |
| 5     | 97% |
| 15    | 99% |

The model achieves high structured translation accuracy due to grammar-expanded dataset design.

---

## 📁 Project Structure
