🏥 Healthcare Assistant

This project is an AI-powered retrieval-based healthcare assistant. It takes symptoms as input from the user, retrieves the most likely disease, and suggests the appropriate precautions and medicines.

The system uses:
FAISS Vector Database for fast similarity search
RAG (Retrieval-Augmented Generation) pipeline for retrieving relevant knowledge
Sentence-Transformers/paraphrase-MiniLM-L3-v2 for creating embeddings of symptoms, diseases, medicines, and precautions

🚀 Features

Symptom-based disease diagnosis
Provides recommended precautions
Suggests suitable medicines
Fast vector similarity search with FAISS
Lightweight and easy-to-use Streamlit interface


⚙️ Installation & Setup

Clone the repository and run the project with the following steps:

# 1. Clone the repository
git clone https://github.com/hammadalam1/HeatlhCare-Assistant.git

# 2. Navigate into the project directory
cd HeatlhCare-Assistant

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run src/chatbot.py

📊 Dataset

The system uses a custom dataset containing:

Diseases
Symptoms
Precautions
Medicines

These are embedded using sentence-transformers and stored in FAISS for retrieval.

🛠️ Tech Stack

Python
Streamlit – for the UI
FAISS (Facebook AI Similarity Search) – for vector search
RAG (Retrieval-Augmented Generation) – for structured retrieval
Sentence-Transformers/paraphrase-MiniLM-L3-v2 – for embeddings

📌 Project Structure
HeatlhCare-Assistant/
│── data/                     # Dataset files (symptoms, medicines, precautions)
│── src/                      # Source code
│   └── chatbot.py             # Main Streamlit application
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation


💡 Disclaimer: This project is for educational and research purposes only. It should not replace professional medical advice.