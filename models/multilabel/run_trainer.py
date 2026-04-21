import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os

from tqdm import tqdm
from transformers import AutoModel, BertConfig
from data.data_preparation import train_dataload, tokenized_intent, listof_intent, val_dataload
from models.multilabel.bert_model import BertEmbedding
from config import opt

def train():        
    """Main training pipeline"""
    
    print("Loading model and optimizer...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    jumlah_inten = len(listof_intent)
    config = BertConfig.from_pretrained(opt.MODEL_NAME)
    model = BertEmbedding(config, num_labels=jumlah_inten)

    model = model.to(device)
    print("Model loaded successfully in: ", device)

    # optimizer, criterion
    optimizer = AdamW(model.parameters(), weight_decay=0.01, lr=opt.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

    intent_ids = tokenized_intent['input_ids'].to(device)
    intent_mask = tokenized_intent['attention_mask'].to(device)
    best_val_f1 = 0

    # Start training
    for epoch in range(opt.epochs):
        print("====== epoch %d / %d: ======"% (epoch+1, opt.epochs))

        # Training Phase
        model.train()
        total_train_loss = 0
        all_predict = []
        all_target = []

        # progress bar
        train_loop = tqdm(train_dataload, desc="Training", leave=False)

        for step, batch in enumerate(train_loop):
            optimizer.zero_grad()

            # ambil data pertanyaan
            pertanyaan_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # ambil data intent
            intent_ids = tokenized_intent['input_ids'].to(device)
            intent_mask = tokenized_intent['attention_mask'].to(device)

            # data yang telah diambil dimasukkan ke LABAN
            logits = model(
                utterance_ids=pertanyaan_ids,
                utterance_mask=attention_mask,
                label_ids=intent_ids,
                Label_mask=intent_mask,
            )

            # hitung error dan koreksi
            train_loss = criterion(logits, labels)
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= opt.thresold).astype(int)
            targets = labels.detach().cpu().numpy()

            all_predict.extend(preds)
            all_target.extend(targets)

            # update progress bar
            train_loop.set_postfix({'train_loss': train_loss.item()})

        print('Average train loss: {:.4f} '.format(total_train_loss / len(train_dataload)))
        
        f1 = f1_score(all_target, all_predict, average='micro')
        print(f'F1 = {f1:.4f}')

    # validation phase
    model.eval()
    val_loss_total = 0
    vall_predict = []
    vall_target = []

    #progress bar validation
    val_loop = tqdm(val_dataload, desc="Validation", leave=False)

    with torch.no_grad():
        for step, batch in enumerate(val_loop):
            pertanyaan_ids_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            intent_ids = tokenized_intent['input_ids'].to(device)
            intent_mask = tokenized_intent['attention_mask'].to(device)

            logits = model(
                utterance_ids=pertanyaan_ids_ids,
                utterance_mask=attention_mask,
                label_ids=intent_ids,
                Label_mask=intent_mask,
            )

            val_loss = criterion(logits, labels)
            val_loss_total += val_loss.item()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= opt.thresold).astype(int)
            targets = labels.detach().cpu().numpy()

            vall_predict.extend(preds)
            vall_target.extend(targets)

            val_loop.set_postfix({'val_loss': val_loss.item()})

    # hitung skor validasi
    val_avg_loss = val_loss_total / len(val_dataload)
    val_f1 = f1_score(vall_target, vall_predict, average='micro')

    print('Average validation loss: {:.4f} '.format(val_avg_loss))
    print(f'Validation F1 = {val_f1:.4f}')

    # menyimpan model
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    if val_f1 > best_val_f1:  # Simpan model jika F1 validasi lebih besar dari sebelumnya
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'checkpoint/IndoBERT_multi_label.pt')
        print('Model saved to checkpoint/IndoBERT_multi_label.pt')
    
if __name__ == '__main__':
    train()