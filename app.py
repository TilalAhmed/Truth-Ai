"""
TruthScan AI - Fake News Detection & Verification System
Flask Backend with LLM + ML Hybrid Architecture
Production-Ready for Hugging Face Spaces
"""

from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv
import pickle
import os
import re
import string
import sqlite3
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import trafilatura
from newspaper import Article
from duckduckgo_search import DDGS  # Fixed import name to match official package
from concurrent.futures import ThreadPoolExecutor
import logging
import sys

# ── LOGGING SETUP ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ])
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ── ENVIRONMENT VALIDATION ─────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY not found. LLM features will be unavailable.")
else:
    logger.info("✓ GROQ_API_KEY loaded successfully")

# ── FLASK APP INITIALIZATION ───────────────────────────
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize Groq client (if available)
try:
    client = Groq(api_key=GROQ_API_KEY)
    groq_available = True
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    client = None
    groq_available = False

# ── DATABASE SETUP (HF Spaces Compatible) ──────────────
# Use /tmp for ephemeral storage on HF Spaces, or current dir if running locally
if os.path.exists('/tmp'):
    DB_PATH = '/tmp/history.db'
else:
    DB_PATH = 'history.db'
logger.info(f"Database path: {DB_PATH}")

def get_db_connection():
    """Get database connection with proper error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def init_db():
    """Initialize database schema"""
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("Cannot initialize database: connection failed")
            return
                
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                headline TEXT,
                prediction TEXT,
                confidence REAL,
                model_used TEXT,
                date TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

def save_to_db(headline, prediction, confidence, model_used):
    """Save prediction to database with error handling"""
    try:
        conn = get_db_connection()
        if not conn:
            logger.warning("Could not save to database: connection failed")
            return False
                
        cursor = conn.cursor()
        date = datetime.now().strftime("%d %b %Y %H:%M")
                
        # Truncate headline to prevent oversized entries
        headline_safe = headline[:500] if headline else "Unknown"
                
        cursor.execute('''
            INSERT INTO predictions
            (headline, prediction, confidence, model_used, date)
            VALUES (?, ?, ?, ?, ?)
        ''', (headline_safe, prediction, round(confidence, 2), model_used, date))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")
        return False

def get_history():
    """Fetch prediction history with error handling"""
    try:
        conn = get_db_connection()
        if not conn:
            return []
                
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, headline, prediction, confidence, model_used, date
            FROM predictions
            ORDER BY id DESC
            LIMIT 50
        ''')
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        return []

def clear_all_history():
    """Clear prediction history with error handling"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
                
        cursor = conn.cursor()
        cursor.execute('DELETE FROM predictions')
        conn.commit()
        conn.close()
        logger.info("History cleared")
        return True
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        return False

# ── URL SCRAPER (Optimized Pipeline) ───────────────────
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.google.com/',
}
REQUEST_TIMEOUT = 5  
MIN_WORD_COUNT = 20  

def try_trafilatura(url):
    """Extract article using Trafilatura"""
    try:
        downloaded = trafilatura.fetch_url(url, timeout=REQUEST_TIMEOUT)
        if not downloaded:
            logger.debug(f"Trafilatura: No content downloaded from {url}")
            return None
                
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if text and len(text.split()) >= MIN_WORD_COUNT:
            logger.info(f"✓ Trafilatura succeeded for {url}")
            return text.strip()
    except Exception as e:
        logger.debug(f"Trafilatura failed: {e}")
    return None

def try_newspaper(url):
    """Extract article using Newspaper3k"""
    try:
        article = Article(url, timeout=REQUEST_TIMEOUT)
        article.download()
        article.parse()
        text = article.text
                
        if article.title:
            text = f"{article.title}\n\n{text}"
                
        if text and len(text.split()) >= MIN_WORD_COUNT:
            logger.info(f"✓ Newspaper3k succeeded for {url}")
            return text.strip()
    except Exception as e:
        logger.debug(f"Newspaper3k failed: {e}")
    return None

def try_beautifulsoup(html_content):
    """Extract article from raw HTML using BeautifulSoup"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
                
        # Remove noise tags
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'iframe']):
            tag.decompose()
                
        # Extract title
        title = ''
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            title = og_title['content'].strip()
        elif soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
                
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        good_paragraphs = [
            p.get_text(strip=True) for p in paragraphs 
            if len(p.get_text(strip=True).split()) > 5
        ]
        article_text = ' '.join(good_paragraphs)
        full_text = f"{title}\n\n{article_text}".strip()
                
        if len(full_text.split()) >= MIN_WORD_COUNT:
            logger.info(f"✓ BeautifulSoup succeeded")
            return full_text
    except Exception as e:
        logger.debug(f"BeautifulSoup failed: {e}")
    return None

def fetch_article_from_url(url):
    """
    Fetch article from URL with fallback pipeline:
    1. Trafilatura
    2. Newspaper3k
    3. BeautifulSoup
    """
    logger.info(f"Fetching article from: {url}")
        
    html_content = None
        
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
        
    try:
        response = requests.get(
            url, 
            headers=HEADERS, 
            timeout=REQUEST_TIMEOUT, 
            allow_redirects=True
        )
                
        if response.status_code == 404:
            return None, "Article not found (404). Please check the URL."
        elif response.status_code == 403:
            return None, "Access denied (403). This website may block automated scraping."
        elif response.status_code >= 400:
            return None, f"Server error ({response.status_code}). Please try another URL."
                
        if response.status_code == 200:
            html_content = response.content
            logger.debug(f"HTML downloaded: {len(html_content)} bytes")
    except requests.exceptions.Timeout:
        return None, "Request timed out. Website is too slow. Please try another URL."
    except requests.exceptions.ConnectionError:
        return None, "Connection failed. Check internet or website availability."
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request exception: {e}")
        return None, "Could not connect to the website. Please try again."
    except Exception as e:
        logger.error(f"Unexpected error fetching URL: {e}")
        return None, "Unexpected error. Please try another URL."
        
    text = try_trafilatura(url)
    if text:
        return text, None
        
    text = try_newspaper(url)
    if text:
        return text, None
        
    if html_content:
        text = try_beautifulsoup(html_content)
        if text:
            return text, None
        
    logger.warning(f"All extraction methods failed for {url}")
    return None, (
        "Could not extract article from this website. "
        "Please copy-paste the article text manually instead."
    )

# ── LOAD MODELS ────────────────────────────────────────
MODEL_DIR = "models"
MODEL_FILES = {
    "Decision Tree": "DT.pkl",
    "Gradient Boosting": "GB.pkl",
    "Logistic Regression": "LR.pkl",
    "Naive Bayes": "NB.pkl",
    "Linear SVM": "SVC.pkl"
}

def load_models():
    """Load ML models and vectorizer with error handling"""
    models = {}
    vectorizer = None
        
    vec_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    if os.path.exists(vec_path):
        try:
            with open(vec_path, "rb") as f:
                vectorizer = pickle.load(f)
            logger.info("✓ Vectorizer loaded")
        except Exception as e:
            logger.error(f"Failed to load vectorizer: {e}")
    else:
        logger.warning(f"Vectorizer not found at {vec_path}")
        
    for name, file in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, file)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    models[name] = pickle.load(f)
                logger.info(f"✓ {name} model loaded")
            except Exception as e:
                logger.error(f"Failed to load {name} model: {e}")
        else:
            logger.warning(f"{name} model not found at {path}")
        
    if not models:
        logger.error("❌ No models loaded! ML predictions will fail.")
        
    return models, vectorizer

models, vectorizer = load_models()

def clean_text(text):
    """Clean and normalize text for ML"""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def get_ml_prediction(text, model_name):
    """Get ML model prediction with fallback"""
    if not models or not vectorizer:
        logger.warning("ML models not available")
        return "UNKNOWN", 0.0
        
    if model_name not in models:
        logger.warning(f"Model '{model_name}' not found, using first available")
        model_name = list(models.keys())[0] if models else None
        
    if not model_name:
        return "UNKNOWN", 0.0
        
    try:
        model = models[model_name]
        cleaned = clean_text(text)
                
        if not cleaned.strip():
            logger.warning("Text too short after cleaning")
            return "UNKNOWN", 0.0
                
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        label = "REAL" if str(pred).lower() in ['1', 'true', 'real'] else "FAKE"
                
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            confidence = round(max(proba) * 100, 2)
        elif hasattr(model, "decision_function"):
            import numpy as np
            score = model.decision_function(vec)[0]
            confidence = round(min(99.9, 50 + abs(float(score)) * 15), 2)
        else:
            confidence = 75.0
                
        return label, confidence
    except Exception as e:
        logger.error(f"ML prediction error: {e}")
        return "UNKNOWN", 0.0

def get_search_queries(text):
    """Generate search queries for fact-checking with error handling"""
    if not groq_available:
        logger.warning("Groq not available, using fallback queries")
        sentences = [s.strip() for s in text.split('.') if len(s.split()) >= 8]
        return sentences[:2] if sentences else [text[:100]]
        
    try:
        prompt = f"""Read the following text. Output up to 2 short web search queries (5-8 words each) to verify its key claims.
Output ONLY the queries, one per line, no numbering or bullets.
Text:{text[:800]}"""
                
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1
        )
                
        raw = response.choices[0].message.content.strip()
        queries = [
            q.strip().strip('"').lstrip('-•123. ') 
            for q in raw.split('\n') if q.strip()
        ]
        queries = [q for q in queries if len(q.split()) >= 2][:2]
                
        if queries:
            logger.info(f"Generated {len(queries)} search queries")
            return queries
    except Exception as e:
        logger.warning(f"Query generation failed: {e}")
        
    for line in text.strip().split('\n'):
        line = line.strip()
        if len(line.split()) >= 5 and not line.lower().startswith(('note:', 'subscribe')):
            return [line[:120]]
    return [' '.join(text.strip().split()[:12])[:120]]

def search_for_grounding(query):
    """Search web for fact-checking with error handling"""
    try:
        results = DDGS().text(query, max_results=2)
        if not results:
            logger.debug(f"No search results for: {query}")
            return None
                
        snippets = [f"- {r.get('title', '')}: {r.get('body', '')}" for r in results]
        return '\n'.join(snippets)
    except Exception as e:
        logger.warning(f"Search failed for '{query}': {e}")
        return None

def search_all_claims(queries):
    """Search multiple claims in parallel"""
    results_map = {}
        
    try:
        with ThreadPoolExecutor(max_workers=min(len(queries), 3)) as executor:
            future_to_query = {
                executor.submit(search_for_grounding, q): q 
                for q in queries
            }
            for future in future_to_query:
                query = future_to_query[future]
                try:
                    results_map[query] = future.result(timeout=5)
                except Exception as e:
                    logger.warning(f"Search thread failed: {e}")
                    results_map[query] = None
    except Exception as e:
        logger.error(f"Parallel search failed: {e}")
        
    blocks = []
    for query in queries:
        result = results_map.get(query)
        if result:
            blocks.append(f'Query: "{query}"\n{result}')
            
    return '\n\n'.join(blocks) if blocks else None

def get_llm_analysis(text, ml_label, ml_confidence):
    """Get LLM analysis with graceful fallback"""
    if not groq_available:
        logger.warning("Groq unavailable, returning ML verdict only")
        return f"VERDICT: {ml_label}\nCONFIDENCE: {ml_confidence // 10}\nSUMMARY: ML model prediction (LLM unavailable)\nRECOMMENDATION: Verify with trusted news sources"
        
    try:
        search_queries = get_search_queries(text)
        search_context = search_all_claims(search_queries)
                
        grounding_block = (
            f"CURRENT WEB SEARCH RESULTS:\n{search_context}\n"
            if search_context
            else "\n(No web search results available)\n"
        )
                
        prompt = f"""Traditional ML: {ml_label} ({ml_confidence}%){grounding_block}
Analyze article content:{text[:1200]}
Respond in EXACTLY this format:
VERDICT: [FAKE or REAL]
CONFIDENCE: [1-10]
CLAIM_SCORE: [1-10]
LANGUAGE_SCORE: [1-10]
SOURCE_SCORE: [1-10]
KEY_SOURCE: [Top Decisive Search Result Title, or None]
SUMMARY: [2 sentences max]
SELF_CRITIQUE: [1 sentence max]
RED FLAGS:
- [flag 1]
- [flag 2]
- [flag 3]
RECOMMENDATION: [1 sentence max]"""
                
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1
        )
                
        result = response.choices[0].message.content
        logger.info("✓ LLM analysis successful")
        return result
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return f"""VERDICT: {ml_label}
CONFIDENCE: {ml_confidence // 10}
CLAIM_SCORE: 6
LANGUAGE_SCORE: 6
SOURCE_SCORE: 5
SUMMARY: LLM analysis unavailable due to API error. Using ML model verdict.
RECOMMENDATION: Verify with trusted news sources."""

def extract_llm_verdict(llm_analysis):
    """Extract verdict from LLM response"""
    lines = llm_analysis.upper()
    if "VERDICT: REAL" in lines:
        return "REAL"
    elif "VERDICT: FAKE" in lines:
        return "FAKE"
    return "UNKNOWN"

def extract_llm_confidence(llm_analysis):
    """Extract confidence from LLM response"""
    match = re.search(r'CONFIDENCE:\s*(\d+)', llm_analysis, re.IGNORECASE)
    if match:
        return round(int(match.group(1)) * 10, 1)
    return 75.0

def extract_score(llm_analysis, field):
    """Extract numeric score from LLM response"""
    match = re.search(rf'{field}:\s*(\d+)', llm_analysis, re.IGNORECASE)
    if match:
        return min(10, max(1, int(match.group(1))))
    return None

def extract_field(llm_analysis, field):
    """Extract text field from LLM response"""
    regex = re.compile(rf'{field}:\s*(.+?)(?=\n[A-Z_]{{2,}}:|$)', re.IGNORECASE | re.DOTALL)
    match = regex.search(llm_analysis)
    if match:
        return match.group(1).strip()
    return None

def compute_truth_percent(verdict, claim_score, language_score, source_score):
    """Compute overall truthfulness percentage from sub-scores"""
    scores = [s for s in (claim_score, language_score, source_score) if s is not None]
        
    if not scores:
        return 85.0 if verdict == "REAL" else 15.0
        
    weights = {
        'claim': 0.5,
        'language': 0.2,
        'source': 0.3
    }
        
    weighted_sum = 0
    weight_total = 0
        
    if claim_score is not None:
        weighted_sum += claim_score * weights['claim']
        weight_total += weights['claim']
    if language_score is not None:
        weighted_sum += language_score * weights['language']
        weight_total += weights['language']
    if source_score is not None:
        weighted_sum += source_score * weights['source']
        weight_total += weights['source']
        
    avg_out_of_10 = weighted_sum / weight_total if weight_total else 5
    truth_percent = round((avg_out_of_10 / 10) * 100, 1)
    return min(99.0, max(1.0, truth_percent))

def validate_input(text):
    """Validate user input"""
    if not text:
        return False, "Please provide text or URL"
        
    text = text.strip()
        
    if len(text) < 20:
        return False, "Input too short (minimum 20 characters)"
        
    if len(text) > 50000:
        return False, "Input too long (maximum 50,000 characters)"
        
    return True, ""

# ── ROUTES ─────────────────────────────────────────────
@app.route('/')
def home():
    """Home page"""
    try:
        # Pull dynamic database log records
        history_rows = get_history()
        
        # Format explicitly for Jinja rendering matching index.html row mappings
        history_list = [{
            'id': r['id'],
            'headline': r['headline'],
            'prediction': r['prediction'],
            'confidence': r['confidence'],
            'model_used': r['model_used'],
            'date': r['date']
        } for r in history_rows]
        
        return render_template(
            'index.html', 
            models=list(MODEL_FILES.keys()),
            history=history_list
        )
    except Exception as e:
        logger.error(f"Home route error: {e}")
        return "Server error", 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze news article"""
    try:
        data = request.get_json()
                
        if not data:
            return jsonify({'error': 'Invalid request'}), 400
                
        news_text = data.get('text', '').strip()
        url_input = data.get('url', '').strip()
        model_name = data.get('model', 'Decision Tree')
                
        if url_input:
            fetched_text, error = fetch_article_from_url(url_input)
            if error:
                logger.warning(f"URL fetch error: {error}")
                return jsonify({'error': error}), 400
            news_text = fetched_text
                
        valid, error_msg = validate_input(news_text)
        if not valid:
            return jsonify({'error': error_msg}), 400
                
        logger.info(f"Analyzing with model: {model_name}")
                
        ml_label, ml_confidence = get_ml_prediction(news_text, model_name)
                
        llm_analysis = get_llm_analysis(news_text, ml_label, ml_confidence)
                
        final_verdict = extract_llm_verdict(llm_analysis)
        final_confidence = extract_llm_confidence(llm_analysis)
                
        claim_score = extract_score(llm_analysis, 'CLAIM_SCORE')
        language_score = extract_score(llm_analysis, 'LANGUAGE_SCORE')
        source_score = extract_score(llm_analysis, 'SOURCE_SCORE')
        key_source = extract_field(llm_analysis, 'KEY_SOURCE')
        self_critique = extract_field(llm_analysis, 'SELF_CRITIQUE')
                
        if final_verdict == "UNKNOWN":
            final_verdict = ml_label
            final_confidence = ml_confidence
                
        save_to_db(news_text[:200], final_verdict, final_confidence, model_name)
                
        truth_percent = compute_truth_percent(
            final_verdict, 
            claim_score, 
            language_score, 
            source_score
        )
                
        # Returns perfectly bound JSON values matching index.html target parameters
        return jsonify({
            'ml_label': final_verdict,
            'ml_confidence': final_confidence,
            'llm_analysis': llm_analysis,
            'model_used': model_name,
            'ml_original': ml_label,
            'ml_original_confidence': ml_confidence,
            'claim_score': claim_score,
            'language_score': language_score,
            'source_score': source_score,
            'key_source': key_source if key_source else "None",
            'self_critique': self_critique if self_critique else "None",
            'truth_percent': truth_percent,
            'fake_percent': round(100 - truth_percent, 1)
        })
    except Exception as e:
        logger.error(f"Analyze error: {e}", exc_info=True)
        return jsonify({'error': 'Server error. Please try again.'}), 500

@app.route('/fetch-url', methods=['POST'])
def fetch_url():
    """Fetch and extract article from URL"""
    try:
        data = request.get_json()
                
        if not data:
            return jsonify({'error': 'Invalid request'}), 400
                
        url = data.get('url', '').strip()
                
        if not url:
            return jsonify({'error': 'Please provide a URL'}), 400
                
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
                
        text, error = fetch_article_from_url(url)
                
        if error:
            logger.warning(f"URL extraction failed: {error}")
            return jsonify({'error': error}), 400
                
        return jsonify({'text': text})
    except Exception as e:
        logger.error(f"Fetch URL error: {e}")
        return jsonify({'error': 'Failed to fetch URL'}), 500

@app.route('/history')
def history():
    """Get prediction history"""
    try:
        rows = get_history()
        return jsonify([{
            'id': r['id'],
            'headline': r['headline'],
            'prediction': r['prediction'],
            'confidence': r['confidence'],
            'model_used': r['model_used'],
            'date': r['date']
        } for r in rows])
    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify([]), 500

@app.route('/clear-history', methods=['POST'])
def clear_history_route():
    """Clear prediction history"""
    try:
        success = clear_all_history()
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        return jsonify({'success': False}), 500

@app.route('/health')
def health():
    """Health check endpoint for HF Spaces"""
    return jsonify({
        'status': 'ok',
        'models_loaded': len(models),
        'vectorizer_loaded': vectorizer is not None,
        'groq_available': groq_available,
        'database': 'ok' if get_db_connection() else 'error'
    })

# ── INITIALIZATION ─────────────────────────────────────
init_db()

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("TruthScan AI - Starting Server")
    logger.info("=" * 60)
    logger.info(f"Models loaded: {len(models)}/{len(MODEL_FILES)}")
    logger.info(f"Vectorizer: {'✓' if vectorizer else '✗'}")
    logger.info(f"Groq API: {'✓ Available' if groq_available else '✗ Unavailable'}")
    logger.info("=" * 60)
        
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 7860)),
        debug=False  
    )
