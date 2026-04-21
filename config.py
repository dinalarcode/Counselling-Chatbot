class config:
    BATCH_SIZE = 16
    max_len = 128
    epochs = 12
    LEARNING_RATE = 2e-5
    MODEL_NAME = 'indobenchmark/indobert-base-p1'
    thresold = 0.5
    device = 'cuda'

opt = config()