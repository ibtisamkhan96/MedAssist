# 🏥 MedAssist: AI-Powered Medical Research Assistant

MedAssist is a Retrieval-Augmented Generation (RAG) system designed to answer complex medical queries by retrieving insights from over 20,000 PubMed research papers. It utilizes the Llama 3.3 70B model (via Groq) for high-speed inference and FAISS for efficient vector retrieval.

## 🚀 Features
* **PubMed Data Pipeline:** Automates fetching, cleaning, and chunking of medical literature using Biopython.
* **Vector Search:** Uses `sentence-transformers/all-MiniLM-L6-v2` and FAISS for sub-millisecond similarity search.
* **LLM Integration:** Powered by Groq API (Llama-3.3-70b) to generate evidence-based answers.
* **Interactive UI:** A user-friendly web interface built with Gradio.

## 📂 Project Structure
* `notebooks/`: Contains the main analysis and pipeline code.
* `src/`: Custom Python modules for retrieval logic.
* `data/`: Folder structure for storing raw papers and embeddings (local only).

## ⚙️ Usage
1. Clone the repo.
2. Install requirements: `pip install -r requirements.txt`
3. Run the notebook `notebooks/MedAssist_Project.ipynb` to execute the pipeline and launch the app.