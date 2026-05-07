class config:
    BATCH_SIZE = 32
    max_len = 128
    epochs = 50
    LEARNING_RATE = 1e-5
    MODEL_NAME = 'indobenchmark/indobert-lite-base-p1'
    thresold = 0.5
    device = 'cuda'
    hidden_size = 768

opt = config()