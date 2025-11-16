# Multilingual Query Handler 🌐

A **Real-Time Multilingual Query Handler** using HuggingFace models.  
Detects input language, translates to English, and computes a basic embedding similarity score.

---

## Features

- Supports **50+ languages** (Hindi, Telugu, Spanish, French, Chinese, Arabic, etc.)
- Uses **facebook/mbart-large-50-many-to-one-mmt** for translation
- Uses **sentence-transformers/all-MiniLM-L6-v2** for embeddings
- Clean **Streamlit UI** for easy testing
- Free, local, **no API keys required**

---

## Installation

1. Clone the repo:

```bash
git clone https://github.com/<YOUR_USERNAME>/multilingual-query-handler.git
cd multilingual-query-handler

2) Create a virtual environment and activate it:

python -m venv .venv
.\.venv\Scripts\activate

3)Install dependencies:

pip install -r requirements.txt

4)Run the Streamlit app:

streamlit run app.py