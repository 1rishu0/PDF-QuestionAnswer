# PDF-QuestionAnswer

A Python-based PDF Question-Answering system powered by **Retrieval-Augmented Generation (RAG)**.

This project extracts text from PDFs, chunks it, generates embeddings (using Hugging Face or OpenAI models), stores them in **Qdrant** (an open-source vector database), and performs efficient similarity search to retrieve relevant context. The retrieved context is then used to augment prompts for a Large Language Model, delivering accurate and document-grounded answers.

Perfect for building custom document-specific QA applications.

## Features

- PDF text extraction and intelligent chunking
- Support for Hugging Face or OpenAI embeddings
- Vector storage and fast similarity search with Qdrant
- Full RAG pipeline for reliable question answering
- Interactive Streamlit web interface

## Tech Stack

- Python
- Qdrant (vector database)
- LangChain (RAG orchestration)
- Streamlit (UI)
- Hugging Face Transformers / OpenAI API
- PyPDF2 or similar for PDF processing

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/1rishu0/PDF-QuestionAnswer.git
   cd PDF-QuestionAnswer

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Set up environment variables (required for OpenAI usage):
   Create a .env file in the root directory:
   ```text
   OPENAI_API_KEY=your_openai_api_key_here

5. Run Qdrant using Docker:
   ```bash
   docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

## Demo Screenshots

<div align="center">
  <img src="https://github.com/1rishu0/PDF-QuestionAnswer/blob/main/RAG_PDF_SS.png" alt="Streamlit App Screenshot" width="800"/>
  <br>
  <em>The Streamlit interface for uploading PDFs and asking questions.</em>
</div>

# Usage

## Streamlit Web App


Launch the Interactive Interface:
```bash
streamlit run streamlit_app.py
```

#### Steps:
- Upload your PDF document
- Wait for the document to be processed and indexed
- Start asking questions about the content

### Command-Line Interface

Run the CLI version:
```bash
python main.py --pdf_pdf path/to/your/document.pdf
```

**Steps:**
- The PDF will be loaded and indexed
- Follow the console prompts to ask questions

## Configuration Tips

- For local/free usage: The project supports Hugging Face embedding models (no API key needed)
- For better performance: Use OpenAI embeddings/LLM by setting OPENAI_API_KEY
- Persistent Qdrant storage: The Docker command above mounts a local folder for data persistence

## Project Structure
```text
.
├── streamlit_app.py      # Streamlit web interface
├── main.py               # CLI entry point
├── data_loader.py        # PDF loading, text extraction & chunking
├── vector_db.py          # Qdrant integration (indexing & retrieval)
├── custom_types.py       # Custom type definitions
├── pyproject.toml        # Dependencies (Poetry)
└── README.md             # This file
```

## Contributing

Contributions are welcome! Feel free to:

- Open issues for bugs or feature requests
- Fork the repo and submit pull requests
- Improve documentation or add examples

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.
```text

You can directly copy and paste this content into your repository's `README.md` file. All sections after **Installation** are now fully formatted with proper Markdown headings, lists, code blocks, and bold text for better readability.
```
