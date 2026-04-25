import torch

from config import opt
from data.data_preparation import listof_intent, tokenized_intent
from models.multilabel.bert_model import BertEmbedding
from transformers import AutoTokenizer

class predictor:
    def __init__(self, model_path = 'checkpoint/IndoBERT_multi_label.pt', thresold = 0.5):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device ('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        self.thresold = thresold

        # kerangka model
        jumlah_inten = len(listof_intent)
        self.model = BertEmbedding(opt, num_labels = jumlah_inten)

        # memuat hasil train ke kerangka
        self.model.load_state_dict(torch.load(model_path, map_location = self.device, weights_only = True))
        self.model.to(self.device)

        self.model.eval()

        self.intent_ids = tokenized_intent['input_ids'].to(self.device)
        self.intent_mask = tokenized_intent['attention_mask'].to(self.device)
        self.label_names = listof_intent
    
    def predict(self, text):
        # Tokenisasi input pengguna
        encoding = self.tokenizer(
            text,
            add_special_tokens = True,
            max_length = 128,
            padding='max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        utterance_ids = encoding['input_ids'].to(self.device)
        utterance_mask = encoding['attention_mask'].to(self.device)
        
        # model menebak
        with torch.no_grad():
            logits = self.model(
                utterance_ids = utterance_ids,
                utterance_mask = utterance_mask,
                label_ids = self.intent_ids,
                Label_mask = self.intent_mask
            )

        # bentuk skor menjadi persentase (0.0 - 1.0)
        probz = torch.sigmoid(logits).squeeze().cpu().numpy()

        # evaluasi berdasar kan thresold
        detected_intents = []
        scores ={}

        for hehe, user in enumerate(self.label_names):
            percentage = float(probz[hehe])
            scores[user] = round(percentage, 4)

            # jika menembus thresold
            if percentage >= self.thresold:
                detected_intents.append(user)

        return{
            "pertanyaan": text,
            "inten_terdeteksi": detected_intents,
            "apakah_multilabel": len(detected_intents) > 1,
            "skore": scores
        }
    
# testing
if __name__ == "__main__":
    try:
        classifier = predictor(thresold=0.5)
        print("Ketik 'done' untuk keluar.\n")
        while True:
            pasien = input("Pasien : ")
            if pasien.lower() == 'done':
                break
            hasil = classifier.predict(pasien)

            print(f"Intent terdeteksi : {hasil['inten_terdeteksi']}")
            print(f"Apakah multilabel? : {hasil['apakah_multilabel']}")
            print(f"Skor : {hasil['skore']}")

    except FileNotFoundError:
        print("Model tidak ditemukan. Pastikan file model berada di path yang benar.")