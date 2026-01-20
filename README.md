# NDA Summarizer

Final year academic project for analyzing and summarizing Non-Disclosure Agreements (NDAs) using NLP and LLM-based techniques.

## Features
- Upload NDA documents (PDF / DOCX)
- Extract key clauses and sections
- Generate concise summaries
- Semantic search using embeddings
- User authentication
- Flask-based web interface

## Tech Stack
- Python
- Flask
- ChromaDB (vector database)
- Transformers & Sentence-Transformers
- MySQL
- OpenAI API (for summarization)

## Setup (Local)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
