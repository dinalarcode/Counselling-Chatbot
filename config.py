class config:
    BATCH_SIZE = 16
    max_len = 50
    epochs = 50
    LEARNING_RATE = 2e-5
    MODEL_NAME = 'indobenchmark/indobert-base-p1'  # Production backbone — highest Test F1-Micro (0.9318) in backbone comparison
    # MODEL_NAME = 'indobenchmark/indobert-lite-base-p1'
    # MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
    # MODEL_NAME = 'indolem/indobertweet-base-uncased'  # Previous model — superseded by IndoBERT
    # MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    # MODEL_NAME = 'intfloat/multilingual-e5-small'
    # MODEL_NAME = 'distilbert-base-multilingual-cased'
    thresold = 0.5
    device = 'cuda'
    hidden_size = 768 
    metric_files_name = f'models/multilabel/{MODEL_NAME.replace("/", "-")}_afteraugmented_training_metrics.png'
    EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # ── Chatbot LLM (RAGEngine: response generation + verse reranker) ────
    # Provider: "groq" | "gemini" | "openai"
    CHATBOT_LLM_PROVIDER = "groq"

    # Groq settings (aktif jika CHATBOT_LLM_PROVIDER = "groq")
    GROQ_CHATBOT_MODEL = "llama-3.3-70b-versatile"
    GROQ_API_KEY_ENV = "GROQ_API_KEY"

    # Gemini settings (aktif jika CHATBOT_LLM_PROVIDER = "gemini")
    GEMINI_CHATBOT_MODEL = "gemini-2.5-flash"
    GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

    # OpenAI settings (aktif jika CHATBOT_LLM_PROVIDER = "openai")
    OPENAI_CHATBOT_MODEL = "gpt-5.4-mini"
    OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

    # Shared chatbot parameter
    CHATBOT_TEMPERATURE = 0.7

    # ── RAGAS Judge LLM ──────────────────────────────────────────────────
    # Fixed: selalu Gemini (biggest context, best Indonesian quality)
    # RAGAS_JUDGE_PROVIDER = "gemini"
    # RAGAS_JUDGE_MODEL = "gemini-2.5-flash"
    # RAGAS_JUDGE_API_KEY_ENV = "GEMINI_API_KEY"
    # RAGAS_JUDGE_TEMPERATURE = 0.0   # deterministic judging
    RAGAS_JUDGE_PROVIDER = "openai"
    RAGAS_JUDGE_MODEL = "gpt-4o-mini"          # used by llm_factory() in eval_ragas.py
    RAGAS_JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
    RAGAS_JUDGE_TEMPERATURE = 0.0   # deterministic judging
    RAGAS_TESTSET_SIZE = 30         # Phase 1 sample count + Phase 2 eval cap (eval_ragas.py)

opt = config()