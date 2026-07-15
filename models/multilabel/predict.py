import torch

from config import opt
from data.data_preparation import listof_intent, tokenized_intent
from models.multilabel.bert_model import BertEmbedding
from transformers import AutoTokenizer

# def _print_stage(title):
#     print(f"\n{'=' * 60}\n--- {title} ---\n{'=' * 60}")

class predictor:
    def __init__(self, model_path = 'checkpoint/IndoBERT_multi_label.pt', thresold = 0.5):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device ('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(opt.MODEL_NAME)
        self.thresold = thresold

        # kerangka model
        jumlah_inten = len(listof_intent)
        self.model = BertEmbedding(num_labels = jumlah_inten)

        # memuat hasil train ke kerangka
        self.model.load_state_dict(torch.load(model_path, map_location = self.device, weights_only = True))
        self.model.to(self.device)

        self.model.eval()

        self.intent_ids = tokenized_intent['input_ids'].to(self.device)
        self.intent_mask = tokenized_intent['attention_mask'].to(self.device)
        self.label_names = listof_intent

        # diagnostic logging without touching bert_model.py or duplicating the forward pass.
        self._label_encoder_capture = {}
        self._utterance_encoder_capture = {}
        self.model.bertlabelencoder.register_forward_hook(
            lambda module, inputs, output: self._label_encoder_capture.update(output=output)
        )
        self.model.bert.register_forward_hook(
            lambda module, inputs, output: self._utterance_encoder_capture.update(output=output)
        )
    
    def predict(self, text):
        # Tokenisasi input pengguna
        encoding = self.tokenizer(
            text,
            add_special_tokens = True,
            max_length = opt.max_len,
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
        logits_flat = logits.squeeze().cpu().numpy()

        # evaluasi berdasar kan thresold
        detected_intents = []
        scores ={}

        for hehe, user in enumerate(self.label_names):
            percentage = float(probz[hehe])
            scores[user] = round(percentage, 4)

            # jika menembus thresold
            if percentage >= self.thresold:
                detected_intents.append(user)

        # self._log_inference_stages(text, utterance_ids, utterance_mask, logits_flat, probz)

        return{
            "pertanyaan": text,
            "inten_terdeteksi": detected_intents,
            "apakah_multilabel": len(detected_intents) > 1,
            "skore": scores
        }

    # def _log_inference_stages(self, text, utterance_ids, utterance_mask, logits_flat, probz):
    #     label_pooler = self._label_encoder_capture['output'].pooler_output
    #     utterance_pooler = self._utterance_encoder_capture['output'].pooler_output

    #     # Stage 1: Hasil Encode Label
    #     _print_stage("STAGE 1: Hasil Encode Label")
    #     print(f"Jumlah label   : {label_pooler.shape[0]}")
    #     print(f"Dimensi vektor : {label_pooler.shape[1]}")
    #     print(f"| {'Label':<45} | {'Norm':>8} | {'Mean':>8} |")
    #     print(f"|{'-'*47}|{'-'*10}|{'-'*10}|")
    #     for name, vec in zip(self.label_names, label_pooler):
    #         print(f"| {name:<45} | {vec.norm().item():>8.4f} | {vec.mean().item():>8.4f} |")

    #     # Stage 2: Hasil Encode Input
    #     _print_stage("STAGE 2: Hasil Encode Input")
    #     tokens = self.tokenizer.convert_ids_to_tokens(utterance_ids.squeeze().tolist())
    #     real_len = int(utterance_mask.sum().item())
    #     print(f"Teks masukan     : {text}")
    #     print(f"Jumlah token     : {real_len} (dari max_len={opt.max_len})")
    #     print(f"Token (terpakai) : {tokens[:real_len]}")

    #     # Stage 3: Hasil Vektor Embedding Semantik
    #     _print_stage("STAGE 3: Hasil Vektor Embedding Semantik")
    #     print(f"Dimensi vektor : {tuple(utterance_pooler.shape)}")
    #     print(f"5 nilai pertama: {utterance_pooler.squeeze()[:5].tolist()}")
    #     print(f"Norm           : {utterance_pooler.norm().item():.4f}")

    #     # Stage 4: Hasil Probabilitas (logits mentah sebelum sigmoid)
    #     _print_stage("STAGE 4: Hasil Probabilitas (Logits Mentah)")
    #     print(f"| {'Label':<45} | {'Logit':>10} |")
    #     print(f"|{'-'*47}|{'-'*12}|")
    #     for name, val in zip(self.label_names, logits_flat):
    #         print(f"| {name:<45} | {float(val):>10.4f} |")

    #     # Stage 5: Hasil Sigmoid + penerapan thresold
    #     _print_stage(f"STAGE 5: Hasil Sigmoid (thresold={self.thresold})")
    #     print(f"| {'Label':<45} | {'Sigmoid':>8} | {'Lolos?':<6} |")
    #     print(f"|{'-'*47}|{'-'*10}|{'-'*8}|")
    #     for name, val in zip(self.label_names, probz):
    #         lolos = "LOLOS" if float(val) >= self.thresold else "-"
    #         print(f"| {name:<45} | {float(val):>8.4f} | {lolos:<6} |")

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