# Hybrid RAG System - Flask App

This is a Flask-based web application for a Hybrid Retrieval-Augmented Generation (RAG) system combining dense embeddings (FAISS), sparse retrieval (BM25), and reciprocal rank fusion (RRF).

## Features

- **Dense Retrieval**: Uses sentence transformers with FAISS for semantic search
- **Sparse Retrieval**: Uses BM25 for keyword-based search
- **Reciprocal Rank Fusion**: Combines both methods for improved results
- **Text Generation**: Uses Google's FLAN-T5 model for answer generation
- **Web UI**: Clean, modern Flask interface with real-time results
- **REST API**: JSON endpoints for programmatic access
- **Metadata Tracking**: Each chunk stores URL, title, and unique ID

## Installation

1. **Set up a virtual environment** (if not already done):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. **Install dependencies**:
```bash
pip install flask requests beautifulsoup4 numpy faiss-cpu sentence-transformers transformers rank-bm25
```

If you have a GPU and want to use FAISS-GPU:
```bash
pip install faiss-gpu
```

## Project Structure

```
hybrid-rag-retrieval/
├── flask_app.py                      # Main Flask application
├── hybrid_rag_system.py              # Original Streamlit version (RAG logic)
├── 200_fixed_urls.json               # Wikipedia URLs dataset
├── requirements.txt                  # Python dependencies
├── templates/
│   └── index.html                   # Web UI template
└── README.md                         # This file
```

## Usage

### Option 1: Web UI (Recommended)

1. **Start the Flask server**:
```bash
python flask_app.py
```

2. **Open in browser**:
   - Navigate to `http://127.0.0.1:5000`
   - Click "Initialize System" to load URLs and build indices (this may take 5-10 minutes on first run)
   - Once ready, enter your question and click "Search"

### Option 2: REST API

#### 1. Initialize the System
```bash
curl -X POST http://127.0.0.1:5000/api/initialize \
  -H "Content-Type: application/json"
```

Response:
```json
{
  "status": "success",
  "message": "System initialized with 5000 chunks from 200 URLs"
}
```

#### 2. Check Health
```bash
curl http://127.0.0.1:5000/health
```

Response:
```json
{
  "status": "healthy",
  "system_initialized": true,
  "chunks_loaded": 5000
}
```

#### 3. Query the System
```bash
curl -X POST http://127.0.0.1:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 10}'
```

Response:
```json
{
  "status": "success",
  "query": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence...",
  "retrieved_chunks": [
    {
      "id": "uuid-1",
      "title": "Machine Learning",
      "url": "https://en.wikipedia.org/wiki/Machine_learning",
      "text": "Machine learning chunk text...",
      "score": 0.8234
    }
  ],
  "response_time_seconds": 2.45
}
```

## Configuration

### Chunking Parameters
Edit `flask_app.py` to modify chunking behavior:

```python
parts = chunk_text_with_metadata(text, url, title, chunk_size=300, overlap=50)
```

- `chunk_size`: Number of tokens per chunk (default: 300)
- `overlap`: Token overlap between consecutive chunks (default: 50)
- Satisfies requirement: 200-400 tokens with 50-token overlap

### Model Selection
To use a different model, update `flask_app.py`:

```python
# For dense embeddings:
dense_index, _, dense_model = build_dense_index(chunks, model_name="sentence-transformers/all-mpnet-base-v2")

# For text generation:
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
```

## Performance Notes

- **First initialization**: 5-15 minutes (downloads models, scrapes URLs, builds indices)
- **Subsequent queries**: 2-5 seconds (after initialization)
- **Memory**: ~4-8 GB for full system with 200 URLs
- **GPU acceleration**: Speeds up embeddings generation by 5-10x

## Metadata Structure

Each chunk includes:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",  // Unique UUID
  "url": "https://en.wikipedia.org/wiki/Example",
  "title": "Example Article",
  "text": "Chunk text content..."
}
```

## Troubleshooting

### Issue: "System not initialized" on query
**Solution**: Call `/api/initialize` first to load data and build indices.

### Issue: Slow initialization
**Solution**: This is normal. Models are large (~2GB+). Use a GPU if available, or reduce the number of URLs in `200_fixed_urls.json`.

### Issue: Out of memory
**Solution**: 
- Reduce chunk_size or max_chars in `fetch_text_from_url()`
- Use fewer URLs
- Use `faiss-gpu` for GPU acceleration

### Issue: Model download fails
**Solution**: Ensure internet connection is stable. Models are cached in `~/.cache/huggingface/`. You can manually download:
```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
huggingface-cli download google/flan-t5-base
```

## Switching from Streamlit to Flask

If you were using the Streamlit version (`hybrid_rag_system.py`), here's what changed:

| Aspect | Streamlit | Flask |
|--------|-----------|-------|
| **Interface** | Widget-based (st.text_input, st.button) | HTML/JavaScript UI + REST API |
| **Initialization** | Automatic on startup | Explicit `/api/initialize` call |
| **Performance** | Recompute on every interaction | Cached indices, persistent memory |
| **Scalability** | Single user | Multiple concurrent users |
| **API** | Built-in sharing (ngrok) | REST endpoints |
| **State Management** | Automatic | Manual with locks |

## API Reference

### POST /api/initialize
Initialize the RAG system with Wikipedia URLs and build indices.
- **Parameters**: None
- **Returns**: `{status, message}`

### POST /api/query
Submit a query to the system.
- **Parameters**: 
  - `query` (string): Your question
  - `top_k` (int, optional): Number of chunks to retrieve (default: 10)
- **Returns**: `{status, query, answer, retrieved_chunks, response_time_seconds}`

### GET /health
Check system health and initialization status.
- **Parameters**: None
- **Returns**: `{status, system_initialized, chunks_loaded}`

### GET /
Serve the web UI.

## License

This project is for educational purposes.

## References

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [BM25 (Rank-BM25)](https://github.com/dorianbrown/rank_bm25)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Flask Documentation](https://flask.palletsprojects.com/)
