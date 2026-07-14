import os


class config:
    BATCH_SIZE = 16
    max_len = 50
    epochs = 50
    LEARNING_RATE = 2e-5
    MODEL_NAME = 'indobenchmark/indobert-base-p1'  # Production backbone — highest Test F1-Micro (0.9318) in backbone comparison
    # MODEL_NAME = 'indolem/indobertweet-base-uncased'  # Previous model — superseded by IndoBERT
    # MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    # MODEL_NAME = 'distilbert-base-multilingual-cased'
    thresold = 0.5
    device = 'cuda'
    hidden_size = 768 
    metric_files_name = f'models/multilabel/{MODEL_NAME.replace("/", "-")}_afteraugmented_training_metrics.png'
    EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Chatbot LLM groq / gemini / openai
    CHATBOT_LLM_PROVIDER = "gemini"

    # if groq
    GROQ_CHATBOT_MODEL = "llama-3.3-70b-versatile"
    # GROQ_CHATBOT_MODEL = "llama-3.1-8b-instant"
    GROQ_API_KEY_ENV = "GROQ_API_KEY"

    # if gemini
    GEMINI_CHATBOT_MODEL = "gemini-2.5-flash"
    GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

    # if openai
    OPENAI_CHATBOT_MODEL = "gpt-5.4-mini"
    OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

    # Shared chatbot parameter
    CHATBOT_TEMPERATURE = 0.7

    # RAGAS Judge LLM
    # RAGAS_JUDGE_PROVIDER = "gemini"
    # RAGAS_JUDGE_MODEL = "gemini-2.5-flash"
    # RAGAS_JUDGE_API_KEY_ENV = "GEMINI_API_KEY"
    # RAGAS_JUDGE_TEMPERATURE = 0.0   # deterministic judging
    RAGAS_JUDGE_PROVIDER = "openai"
    RAGAS_JUDGE_MODEL = "gpt-4o"          # passed to langchain_openai.ChatOpenAI in ragas_manager/ragas_engine.py
    RAGAS_JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
    RAGAS_JUDGE_TEMPERATURE = 0.0   # deterministic judging
    RAGAS_TESTSET_SIZE = 100        # Phase 1 sample count + Phase 2 eval cap (ragas_manager/ragas_engine.py)

    # Data paths ───────────────────────────────────────────────────────
    # Absolute, anchored on config.py's own location, so every script resolves
    # the same file regardless of the directory it is launched from.
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
    VERSE_DIR    = os.path.join(DATA_DIR, "verse_data")
    QNA_DIR      = os.path.join(DATA_DIR, "qna-intent_data")

    # Verse data
    BIBLE_RAW_CSV         = os.path.join(VERSE_DIR, "alkitab_tb.csv")
    BIBLE_ENRICHED_CSV    = os.path.join(VERSE_DIR, "alkitab_tb_enriched_groq.csv")
    CHAPTER_SUMMARIES_CSV = os.path.join(VERSE_DIR, "chapter_summaries.csv")
    FAISS_BIBLE_INDEX     = os.path.join(VERSE_DIR, "faiss_bible_index")

    # QnA / intent data
    QNA_CSV                = os.path.join(QNA_DIR, "dataset_qna.csv")
    MULTIINTENT_CSV        = os.path.join(QNA_DIR, "dataset_multiintent.csv")
    MULTIINTENT_AUG_CSV    = os.path.join(QNA_DIR, "dataset_multiintent_augmented.csv")
    INTENT_CONTENT_CSV     = os.path.join(QNA_DIR, "intent_content.csv")
    INTENT_CONTENT_AUG_CSV = os.path.join(QNA_DIR, "intent_content_augmented.csv")
    FAISS_QNA_INDEX        = os.path.join(QNA_DIR, "faiss_qna_index")

opt = config()