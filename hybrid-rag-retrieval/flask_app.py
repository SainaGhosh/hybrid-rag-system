import json
import time
import faiss
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bs4 import BeautifulSoup
import requests
import uuid
from threading import Lock

# ---------------------------------------------------------------
# Initialize Flask App
# ---------------------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Global variables for indices and models
chunks = []
dense_index = None
dense_model = None
bm25 = None
tokenizer = None
gen_model = None
data_lock = Lock()  # Thread-safe access to shared resources

print("[INFO] Flask app initialized. Indices not yet loaded.")


# ---------------------------------------------------------------
# Helper Functions (from hybrid_rag_system.py)
# ---------------------------------------------------------------
def chunk_text(text, chunk_size=300, overlap=50):
    """Return list of text chunks split by tokens (whitespace).
    Defaults: 300-token chunks with 50-token overlap.
    """
    words = text.split()
    chunks_list = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks_list.append(chunk)
    return chunks_list


def chunk_text_with_metadata(text, url, title, chunk_size=300, overlap=50):
    """Split text into token chunks and return list of dicts with metadata.
    
    Each chunk dict: {'id': <uuid4>, 'url': url, 'title': title, 'text': chunk_text}
    """
    chunks_list = []
    token_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    for part in token_chunks:
        chunk_id = str(uuid.uuid4())
        chunks_list.append({
            'id': chunk_id,
            'url': url,
            'title': title,
            'text': part
        })
    return chunks_list


def fetch_text_from_url(url, max_chars=20000):
    """Fetch a page and return (title, cleaned_text).
    
    On errors returns ("", "").
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Get title (Wikipedia uses h1#firstHeading)
        title_tag = soup.find('h1', id='firstHeading')
        if title_tag:
            title = title_tag.get_text(strip=True)
        else:
            title = soup.title.string if soup.title else ''

        content_div = soup.find('div', {'class': 'mw-parser-output'})
        if content_div:
            paragraphs = content_div.find_all('p')
        else:
            paragraphs = soup.find_all('p')
        
        text = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
        # Trim excessive whitespace and limit size
        text = " ".join(text.split())
        return title, text[:max_chars]
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return "", ""


def load_urls():
    """Load fixed Wikipedia URLs from JSON file."""
    try:
        with open(r'D:\Bits-MTech\Assignments\conv-ai-assginment-2\hybrid-rag-retrieval\200_fixed_urls.json', "r") as f:
            fixed_urls = json.load(f)["fixed_wiki_urls"]
        return fixed_urls
    except Exception as e:
        print(f"[ERROR] Failed to load URLs: {e}")
        return []


def build_dense_index(chunks_list, model_name="all-MiniLM-L6-v2"):
    """Build FAISS index from list of chunk dicts. Uses the chunk['text'] field."""
    print("[INFO] Building dense index...")
    model = SentenceTransformer(model_name)
    texts = [c['text'] for c in chunks_list]
    embeddings = model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[INFO] Dense index built with {len(texts)} chunks.")
    return index, embeddings, model


def build_sparse_index(chunks_list):
    """Build BM25 index from list of chunk dicts."""
    print("[INFO] Building sparse (BM25) index...")
    texts = [c['text'] for c in chunks_list]
    tokenized = [text.split() for text in texts]
    bm25_index = BM25Okapi(tokenized)
    print(f"[INFO] BM25 index built with {len(texts)} chunks.")
    return bm25_index, tokenized


def dense_retrieve(query, index, model, chunks_list, top_k=10):
    """Retrieve top-k chunks using dense embeddings."""
    q_emb = model.encode([query], convert_to_numpy=True)
    scores, ids = index.search(q_emb, top_k)
    return [(chunks_list[i], float(scores[0][j])) for j, i in enumerate(ids[0])]


def sparse_retrieve(query, bm25_index, chunks_list, top_k=10):
    """Retrieve top-k chunks using BM25."""
    scores = bm25_index.get_scores(query.split())
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(chunks_list[i], float(scores[i])) for i in ranked]


def reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=10):
    """Fuse dense and sparse ranked lists. 
    
    Returns list of (chunk_dict, fused_score).
    """
    scores = {}
    id_to_chunk = {}
    
    # Assign ranks
    for rank, (chunk, _) in enumerate(dense_results):
        cid = chunk['id']
        id_to_chunk[cid] = chunk
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    
    for rank, (chunk, _) in enumerate(sparse_results):
        cid = chunk['id']
        id_to_chunk[cid] = chunk
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    
    # Sort by fused score
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(id_to_chunk[cid], score) for cid, score in fused]


def generate_answer(query, context_chunks, model_name="google/flan-t5-base"):
    """Generate answer using T5 model."""
    try:
        # context_chunks is list of (chunk_dict, score)
        context = "\n".join([c['text'] for c, _ in context_chunks])
        prompt = f"Answer the question based on context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = gen_model.generate(**inputs, max_length=200)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        print(f"[ERROR] Failed to generate answer: {e}")
        return f"Error generating answer: {str(e)}"


# ---------------------------------------------------------------
# Initialization Route
# ---------------------------------------------------------------
@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize/load the RAG system with URLs and build indices."""
    global chunks, dense_index, dense_model, bm25, tokenizer, gen_model
    
    with data_lock:
        try:
            print("[INFO] Starting system initialization...")
            
            # Load models
            print("[INFO] Loading tokenizer and model...")
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            print("[INFO] Models loaded.")
            
            # Load and chunk data
            print("[INFO] Loading URLs...")
            urls = load_urls()
            print(f"[INFO] Loaded {len(urls)} URLs.")
            
            all_chunks = []
            successful_urls = 0
            for idx, url in enumerate(urls):
                if idx % 10 == 0:
                    print(f"[INFO] Processing URL {idx+1}/{len(urls)}...")
                title, text = fetch_text_from_url(url)
                if text:
                    parts = chunk_text_with_metadata(text, url, title, chunk_size=300, overlap=50)
                    all_chunks.extend(parts)
                    successful_urls += 1
            
            print(f"[INFO] Processed {successful_urls} URLs successfully. Total chunks: {len(all_chunks)}")
            
            if not all_chunks:
                return jsonify({'status': 'error', 'message': 'No chunks created from URLs'}), 400
            
            chunks = all_chunks
            
            # Build indices
            dense_index, _, dense_model = build_dense_index(chunks)
            bm25, _ = build_sparse_index(chunks)
            
            return jsonify({
                'status': 'success',
                'message': f'System initialized with {len(chunks)} chunks from {successful_urls} URLs'
            })
        except Exception as e:
            print(f"[ERROR] Initialization failed: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500


# ---------------------------------------------------------------
# Query Endpoint
# ---------------------------------------------------------------
@app.route('/api/query', methods=['POST'])
def query_rag():
    """Accept a query and return generated answer + retrieved chunks."""
    global chunks, dense_index, dense_model, bm25
    
    if not chunks:
        return jsonify({'status': 'error', 'message': 'System not initialized. Call /api/initialize first.'}), 400
    
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        
        if not query_text:
            return jsonify({'status': 'error', 'message': 'Query cannot be empty'}), 400
        
        top_k = data.get('top_k', 10)
        
        start_time = time.time()
        
        with data_lock:
            # Retrieve
            dense_results = dense_retrieve(query_text, dense_index, dense_model, chunks, top_k=top_k)
            sparse_results = sparse_retrieve(query_text, bm25, chunks, top_k=top_k)
            rrf_results = reciprocal_rank_fusion(dense_results, sparse_results, top_n=5)
            
            # Generate
            answer = generate_answer(query_text, rrf_results)
        
        elapsed = time.time() - start_time
        
        # Format results
        retrieved_chunks = []
        for chunk, score in rrf_results:
            retrieved_chunks.append({
                'id': chunk['id'],
                'title': chunk['title'],
                'url': chunk['url'],
                'text': chunk['text'],
                'score': round(score, 4)
            })
        
        return jsonify({
            'status': 'success',
            'query': query_text,
            'answer': answer,
            'retrieved_chunks': retrieved_chunks,
            'response_time_seconds': round(elapsed, 2)
        })
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ---------------------------------------------------------------
# Web UI Routes
# ---------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    """Serve the main UI page."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    is_ready = len(chunks) > 0
    return jsonify({
        'status': 'healthy',
        'system_initialized': is_ready,
        'chunks_loaded': len(chunks)
    })


# ---------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == '__main__':
    print("[INFO] Starting Flask app on http://127.0.0.1:5000")
    print("[INFO] Use POST /api/initialize to load data and build indices")
    print("[INFO] Use POST /api/query with {'query': '...'} to perform queries")
    app.run(debug=True, host='127.0.0.1', port=5000)
