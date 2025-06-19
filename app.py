import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import warnings

warnings.filterwarnings('ignore')

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ Sentence Transformers tidak tersedia. Menggunakan TF-IDF sebagai fallback.")


# Download required NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')


# Initialize NLTK resources
download_nltk_resources()


# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    try:
        # Try to load the CSV file
        df_products = pd.read_csv('Obat_Bebas dan Bebas_Terbatas - products.csv')

        # Data cleaning
        df_products['title'] = df_products['title'].astype(str).str.lower().str.strip()
        df_products['description'] = df_products['description'].astype(str).str.lower().str.strip()

        # Remove extra whitespace within the text
        df_products['title'] = df_products['title'].apply(lambda x: re.sub(r'\s+', ' ', x))
        df_products['description'] = df_products['description'].apply(lambda x: re.sub(r'\s+', ' ', x))

        # Remove HTML tags and clean text
        df_products['description'] = df_products['description'].apply(lambda x: re.sub(r'<.*?>', '', x))
        df_products['description'] = df_products['description'].apply(lambda x: re.sub(r'[^\w\s\-\.]', ' ', x))

        # Create combined text for better search
        df_products['combined_text'] = df_products['title'] + " " + df_products['description']

        # Extract key medical terms and symptoms
        df_products['medical_keywords'] = df_products['combined_text'].apply(extract_medical_keywords)

        return df_products

    except FileNotFoundError:
        st.error("File 'Obat_Bebas dan Bebas_Terbatas - products.csv' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")
        return None


# Enhanced text preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words('indonesian'))
    # Add medical-specific stop words
    medical_stop_words = {'obat', 'untuk', 'dengan', 'yang', 'dari', 'dan', 'atau', 'pada', 'dalam', 'dapat', 'akan',
                          'ini', 'itu'}
    stop_words.update(medical_stop_words)

    text = re.sub(r'[^-\u007F\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)


# Extract medical keywords and symptoms
def extract_medical_keywords(text):
    # Common medical terms and symptoms in Indonesian
    medical_terms = {
        'sakit': ['nyeri', 'rasa sakit', 'pegal', 'linu'],
        'kepala': ['migrain', 'pusing', 'vertigo', 'sakit kepala'],
        'demam': ['panas', 'fever', 'meriang', 'hangat'],
        'batuk': ['sesak', 'dahak', 'tenggorokan', 'bronkitis'],
        'pilek': ['hidung', 'ingus', 'flu', 'selesma'],
        'perut': ['mual', 'muntah', 'diare', 'sembelit', 'maag', 'lambung'],
        'alergi': ['gatal', 'ruam', 'bentol', 'biduran', 'eksim'],
        'mata': ['merah', 'gatal', 'berair', 'konjungtivitis'],
        'kulit': ['jerawat', 'eksim', 'dermatitis', 'iritasi'],
        'otot': ['kram', 'kejang', 'tegang', 'nyeri otot'],
        'sendi': ['arthritis', 'rematik', 'bengkak'],
        'tidur': ['insomnia', 'susah tidur', 'gelisah'],
        'stress': ['cemas', 'depresi', 'tegang', 'khawatir']
    }

    keywords = []
    text_lower = text.lower()

    for category, terms in medical_terms.items():
        if any(term in text_lower for term in terms):
            keywords.append(category)

    return ' '.join(keywords)


# Load Sentence Transformer model
@st.cache_resource
def load_sentence_transformer():
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    try:
        # Use multilingual model that works well with Indonesian
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        return model
    except Exception as e:
        st.warning(f"Gagal memuat Sentence Transformer: {e}")
        return None


# Initialize embeddings
@st.cache_data
def create_embeddings(_df_products, _model):
    if _df_products is None or _model is None:
        return None

    # Create enhanced text for embedding
    enhanced_texts = []
    for _, row in _df_products.iterrows():
        # Combine title, description, and medical keywords with proper weighting
        enhanced_text = f"{row['title']} {row['title']} {row['description']} {row['medical_keywords']} {row['medical_keywords']}"
        enhanced_texts.append(enhanced_text)

    # Generate embeddings
    embeddings = _model.encode(enhanced_texts, show_progress_bar=False)
    return embeddings


# Initialize TF-IDF as fallback
@st.cache_resource
def initialize_tfidf(_df_products):
    if _df_products is None:
        return None, None

    _df_products['preprocessed_text'] = _df_products['combined_text'].apply(preprocess_text)

    # Enhanced TF-IDF with better parameters
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Include bigrams
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(_df_products['preprocessed_text'])

    return tfidf_vectorizer, tfidf_matrix


# Enhanced similarity calculation with multiple techniques
def calculate_similarity_advanced(user_input, df_products, model=None, embeddings=None, tfidf_vectorizer=None,
                                  tfidf_matrix=None):
    if df_products is None:
        return []

    results = []

    # Method 1: Sentence Transformer (if available)
    if model is not None and embeddings is not None:
        # Enhance user input with medical keyword extraction
        enhanced_input = user_input + " " + extract_medical_keywords(user_input)
        user_embedding = model.encode([enhanced_input])

        # Calculate cosine similarity
        similarities = cosine_similarity(user_embedding, embeddings)[0]

        for i, score in enumerate(similarities):
            results.append({
                'index': i,
                'semantic_score': score,
                'title': df_products.iloc[i]['title'],
                'description': df_products.iloc[i]['description'],
                'medical_keywords': df_products.iloc[i]['medical_keywords']
            })

    # Method 2: Enhanced TF-IDF (fallback or combination)
    if tfidf_vectorizer is not None and tfidf_matrix is not None:
        preprocessed_input = preprocess_text(user_input)
        input_vector = tfidf_vectorizer.transform([preprocessed_input])
        tfidf_similarities = cosine_similarity(input_vector, tfidf_matrix)[0]

        if not results:  # If no semantic results, use TF-IDF
            for i, score in enumerate(tfidf_similarities):
                results.append({
                    'index': i,
                    'tfidf_score': score,
                    'title': df_products.iloc[i]['title'],
                    'description': df_products.iloc[i]['description'],
                    'medical_keywords': df_products.iloc[i]['medical_keywords']
                })
        else:  # Combine with semantic scores
            for i, result in enumerate(results):
                result['tfidf_score'] = tfidf_similarities[i]

    # Method 3: Keyword matching boost
    user_keywords = set(extract_medical_keywords(user_input).split())

    for result in results:
        product_keywords = set(result['medical_keywords'].split())
        keyword_overlap = len(user_keywords.intersection(product_keywords))
        keyword_score = keyword_overlap / max(len(user_keywords), 1)
        result['keyword_score'] = keyword_score

    # Calculate combined score
    for result in results:
        semantic_score = result.get('semantic_score', 0)
        tfidf_score = result.get('tfidf_score', 0)
        keyword_score = result.get('keyword_score', 0)

        # Weighted combination
        if semantic_score > 0:
            combined_score = (0.5 * semantic_score + 0.3 * tfidf_score + 0.2 * keyword_score)
        else:
            combined_score = (0.7 * tfidf_score + 0.3 * keyword_score)

        result['combined_score'] = combined_score

    # Sort by combined score
    results.sort(key=lambda x: x['combined_score'], reverse=True)

    # Filter out very low scores
    results = [r for r in results if r['combined_score'] > 0.05]

    return results


# Query expansion for better results
def expand_query(user_input):
    # Medical term synonyms in Indonesian
    synonyms = {
        'sakit kepala': ['migrain', 'pusing', 'nyeri kepala', 'vertigo'],
        'demam': ['panas', 'meriang', 'fever'],
        'batuk': ['sesak', 'batuk kering', 'batuk berdahak'],
        'pilek': ['flu', 'hidung tersumbat', 'selesma'],
        'sakit perut': ['mual', 'nyeri lambung', 'maag', 'sakit maag'],
        'alergi': ['gatal', 'ruam', 'biduran', 'eksim'],
        'diare': ['mencret', 'buang air besar cair'],
        'sembelit': ['susah bab', 'konstipasi'],
        'insomnia': ['susah tidur', 'tidak bisa tidur'],
        'stress': ['cemas', 'gelisah', 'tegang']
    }

    expanded_terms = [user_input]
    user_lower = user_input.lower()

    for term, syns in synonyms.items():
        if term in user_lower:
            expanded_terms.extend(syns)

    return ' '.join(expanded_terms)


# Streamlit App
def main():
    st.set_page_config(
        page_title="Sistem Rekomendasi Obat Advanced",
        page_icon="💊",
        layout="wide"
    )

    st.title("💊 Sistem Rekomendasi Obat Advanced")
    st.markdown("*Menggunakan AI dan Machine Learning untuk rekomendasi yang lebih akurat*")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("ℹ️ Informasi Sistem")

    # Show which methods are available
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        st.sidebar.success("✅ Semantic Search (Sentence Transformers)")
    else:
        st.sidebar.warning("⚠️ Semantic Search tidak tersedia")

    st.sidebar.success("✅ TF-IDF Enhanced")
    st.sidebar.success("✅ Medical Keyword Matching")
    st.sidebar.success("✅ Query Expansion")

    st.sidebar.markdown("""
    **Teknik yang Digunakan:**
    - **Semantic Search**: Memahami makna dan konteks
    - **TF-IDF Enhanced**: Pencarian berdasarkan kata kunci
    - **Medical Keywords**: Pencocokan istilah medis
    - **Query Expansion**: Memperluas pencarian dengan sinonim
    - **Hybrid Scoring**: Kombinasi multiple teknik

    **Cara Penggunaan:**
    1. Masukkan keluhan dengan detail
    2. Sistem akan menganalisis dengan multiple teknik
    3. Hasil diurutkan berdasarkan relevansi tertinggi
    """)

    # Load data
    with st.spinner("Memuat data obat..."):
        df_products = load_and_preprocess_data()

    if df_products is None:
        st.error("Tidak dapat memuat data obat.")
        return

    # Load models
    with st.spinner("Memuat model AI..."):
        sentence_model = load_sentence_transformer() if SENTENCE_TRANSFORMERS_AVAILABLE else None
        embeddings = create_embeddings(df_products, sentence_model) if sentence_model else None
        tfidf_vectorizer, tfidf_matrix = initialize_tfidf(df_products)

    # Display system status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Obat", len(df_products))
    with col2:
        st.metric("Semantic Search", "✅" if sentence_model else "❌")
    with col3:
        st.metric("TF-IDF", "✅" if tfidf_vectorizer else "❌")
    with col4:
        st.metric("Status", "Siap")

    st.markdown("---")

    # Main interface
    st.header("🔍 Cari Rekomendasi Obat")

    # Check for query parameter
    query_param = st.query_params.get("query", "")

    # User input with better help text
    user_input = st.text_input(
        "Masukkan keluhan Anda dengan detail:",
        value=query_param,
        placeholder="Contoh: sakit kepala sebelah kiri, demam tinggi disertai batuk, sakit perut mual muntah",
        help="Semakin detail keluhan Anda, semakin akurat rekomendasi yang diberikan"
    )

    # Advanced options
    with st.expander("⚙️ Pengaturan Lanjutan"):
        col1, col2 = st.columns(2)
        with col1:
            num_recommendations = st.slider(
                "Jumlah rekomendasi:",
                min_value=1, max_value=15, value=8
            )
            use_query_expansion = st.checkbox("Gunakan Query Expansion", value=True)

        with col2:
            min_score_threshold = st.slider(
                "Threshold minimum score:",
                min_value=0.0, max_value=1.0, value=0.1, step=0.05
            )
            show_scores = st.checkbox("Tampilkan detail scoring", value=False)

    # Search button
    if st.button("🔍 Cari Rekomendasi", type="primary"):
        if user_input.strip():
            with st.spinner("Menganalisis keluhan dengan AI..."):
                # Expand query if enabled
                search_input = expand_query(user_input) if use_query_expansion else user_input

                # Calculate similarity with advanced methods
                results = calculate_similarity_advanced(
                    search_input, df_products, sentence_model, embeddings,
                    tfidf_vectorizer, tfidf_matrix
                )

                # Filter by threshold
                results = [r for r in results if r['combined_score'] >= min_score_threshold]

                if results:
                    st.success(f"Ditemukan {len(results)} rekomendasi untuk '{user_input}'")

                    if use_query_expansion and search_input != user_input:
                        st.info(f"🔍 Query diperluas menjadi: {search_input}")

                    st.markdown("---")

                    # Display recommendations
                    st.header("📋 Rekomendasi Obat")

                    for i, result in enumerate(results[:num_recommendations]):
                        score_percentage = result['combined_score'] * 100

                        # Determine confidence level
                        if score_percentage >= 70:
                            confidence = "🟢 Sangat Sesuai"
                            confidence_color = "success"
                        elif score_percentage >= 50:
                            confidence = "🟡 Cukup Sesuai"
                            confidence_color = "warning"
                        elif score_percentage >= 30:
                            confidence = "🟠 Kemungkinan Sesuai"
                            confidence_color = "info"
                        else:
                            confidence = "🔴 Kurang Sesuai"
                            confidence_color = "error"

                        with st.expander(f"#{i + 1} - {result['title'].title()}", expanded=(i < 3)):
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                st.write(f"**Nama Obat:** {result['title'].title()}")
                                st.write(f"**Deskripsi:** {result['description']}")

                                if result['medical_keywords']:
                                    st.write(f"**Kategori:** {result['medical_keywords']}")

                            with col2:
                                st.metric("Tingkat Kesesuaian", f"{score_percentage:.1f}%")
                                getattr(st, confidence_color)(confidence)

                                if show_scores:
                                    st.write("**Detail Scoring:**")
                                    if 'semantic_score' in result:
                                        st.write(f"Semantic: {result['semantic_score']:.3f}")
                                    if 'tfidf_score' in result:
                                        st.write(f"TF-IDF: {result['tfidf_score']:.3f}")
                                    st.write(f"Keywords: {result['keyword_score']:.3f}")

                else:
                    st.warning(
                        "⚠️ Tidak ditemukan obat yang sesuai dengan keluhan Anda. Coba gunakan kata kunci yang berbeda atau konsultasikan dengan dokter.")
        else:
            st.warning("Harap masukkan keluhan Anda terlebih dahulu.")

    # Sample queries with more detailed examples
    st.markdown("---")
    st.header("💡 Contoh Pencarian Detail")

    sample_queries = [
        "sakit kepala migrain sebelah kiri",
        "demam tinggi anak disertai batuk",
        "sakit perut mual muntah setelah makan",
        "batuk kering tidak berdahak",
        "alergi kulit gatal merah bentol",
        "susah tidur insomnia stress",
        "diare mencret lebih dari 3 hari",
        "sakit gigi berlubang nyeri berdenyut",
        "mata merah berair gatal alergi",
        "nyeri otot setelah olahraga"
    ]

    cols = st.columns(2)
    for i, query in enumerate(sample_queries):
        with cols[i % 2]:
            if st.button(query, key=f"sample_{i}"):
                st.query_params["query"] = query
                st.rerun()

    # Enhanced disclaimer
    st.markdown("---")
    st.markdown("""
    **⚠️ Peringatan Penting:**
    - Sistem ini menggunakan AI untuk memberikan rekomendasi berdasarkan analisis teks
    - Hasil rekomendasi bukan diagnosis medis dan tidak menggantikan konsultasi dokter
    - Selalu konsultasikan dengan dokter atau apoteker sebelum menggunakan obat
    - Baca aturan pakai, dosis, dan kontraindikasi sebelum mengonsumsi obat
    - Jika gejala memburuk atau tidak membaik, segera konsultasikan ke dokter
    - Untuk kondisi serius atau darurat medis, segera cari bantuan medis profesional
    """)


if __name__ == "__main__":
    main()