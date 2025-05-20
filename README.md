# 🐔 Poultry Disease Classification & QA Assistant

A hybrid deep learning and retrieval-based system for early detection and diagnosis of poultry diseases using image-based classification and a context-aware question-answering chatbot.

## 🧠 Project Overview

This project combines computer vision and retrieval-augmented generation (RAG) to:

- Classify poultry diseases from fecal matter images using a CNN + CBAM (Convolutional Block Attention Module) model.
- Provide real-time, disease-specific answers using a Llama-based QA bot integrated with FAISS.
- Deliver results through a user-friendly web interface.

The goal is to support poultry farmers and veterinarians in early disease detection and intervention.

---

## 📌 Key Features

### 🔍 Disease Classification
- **Hybrid CNN Model with CBAM**: Improves spatial attention and feature prioritization.
- **Recognizes 4 Categories**:
  - `Cocci`: Coccidiosis infection
  - `Healthy`: No signs of disease
  - `NCD`: Newcastle Disease (viral)
  - `Salmonella`: Bacterial infection

### 🤖 RAG-Based QA Bot
- **Llama model** for natural language understanding  
- **FAISS-powered** vector store for fast and relevant knowledge retrieval  
- Offers **real-time answers** to poultry disease-related questions

### 🌐 Web Interface
- Upload fecal images and receive instant predictions
- Interact with a disease-specific chatbot
- Built for accessibility by farmers, vets, and researchers

---

## 🛠️ Tech Stack

| Component        | Technology              |
|------------------|--------------------------|
| Model Architecture | CNN + CBAM              |
| QA System         | RAG (Llama + FAISS)      |
| Backend           | Python, FastAPI   |
| Frontend          | HTML/CSS/JS |
| ML Libraries      | PyTorch / TensorFlow     |
| Vector Store      | FAISS                    |

---

