class config:
    BATCH_SIZE = 16
    max_len = 50
    epochs = 50
    LEARNING_RATE = 2e-5
    # MODEL_NAME = 'indobenchmark/indobert-lite-base-p1'
    # MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
    MODEL_NAME = 'indolem/indobertweet-base-uncased' # sejauh ini paling akurat secara predict
    # MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    # MODEL_NAME = 'intfloat/multilingual-e5-small'
    # MODEL_NAME = 'distilbert-base-multilingual-cased' 
    thresold = 0.5
    device = 'cuda'
    hidden_size = 768 
    metric_files_name = f'models/multilabel/{MODEL_NAME.replace("/", "-")}_afteraugmented_training_metrics.png'

opt = config()