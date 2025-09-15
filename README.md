# Research Abstract Generator (Llama Stack)

A **Llama Stack–backed application** to facilitate **abstract generation** for academic research papers.  
Built during the **8VC x Meta x HackDuke CodeFest**.

- **Model:** `llama-3-8b-bnb-4bit` (8 billion parameters)  
- **Fine-tuning Method:** LoRA / QLoRA using the [Unsloth library](https://github.com/unslothai/unsloth?tab=readme-ov-file)  
- **Frontend:** [Streamlit](https://streamlit.io/)  
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) with **Llama Stack orchestration**

👉 Fine-tuned model is available on HuggingFace:  
[**LoganChu/research-abstract-llama-gguf**](https://huggingface.co/LoganChu/research-abstract-llama-gguf)

---

## 🚀 Features
- Research abstract summarization from academic text.  
- Conversational chatbot capabilities powered by Llama3.  
- Optimized with **bnb-4bit quantization** for reduced memory and faster inference.  

---

## 🧩 Model Card: `abstract-research-llama-gguf`

### 📌 Model Overview
- **Base Model:** Llama3-8B (bnb-4bit quantized)  
- **Fine-Tuning Framework:** Unsloth + HuggingFace Transformers + TRL  
- **Adapters:** LoRA, QLoRA  
- **GPU Used:** Google Colab Free Tesla T4  

The model was trained on a **Kaggle dataset of paper abstracts**, converted to **ShareGPT conversational format**.  

---

### 📖 Uses
- Conversational AI and chatbot applications  
- Summarization of research abstracts  

---

### 📊 Training Data
- Dataset: [Kaggle – ArXiv Paper Abstracts](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts)  
- Data formatting: Converted into `<instruction, output>` pairs using  
  - `to_sharegpt`  
  - `standardize_sharegpt`  

---

### ⚙️ Training Procedure
- **Notebook:** [Unsloth Training Guide](https://github.com/unslothai/unsloth?tab=readme-ov-file)  
- **Batch Size:** 2 (gradient accumulation 4 → effective batch 8)  
- **Max Sequence Length:** 2048 tokens  
- **Epochs/Steps:** 1 epoch (~100 steps demo, scalable further)  
- **Learning Rate:** 1e-4  

---

### 💬 Chat Template
A **custom instruction-following template** was used (similar to Llama3, Alpaca, or ChatML variants) with support for:
- `system`
- `user`
- `assistant`

---

### ⚠️ Limitations
- Performance tuned for **abstract summarization**; may underperform on unrelated tasks.  
- Can **hallucinate** or generate incorrect outputs when prompted **out of domain**.  

---

## 🛠️ Tech Stack
- **Llama Stack** (orchestration)  
- **Streamlit** (frontend UI)  
- **FastAPI** (backend service)  
- **HuggingFace + TRL + Unsloth** (fine-tuning + inference)  

---

## 📜 License
Open for educational and research purposes. Please check dataset and model base licenses before commercial use.  
