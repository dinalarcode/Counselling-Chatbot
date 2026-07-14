import os
import requests
import torch
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config import opt

pd.set_option('display.max_columns', None)

# Toggle: set to True to use the augmented (larger) dataset
USE_AUGMENTED = True
_aug_path = opt.MULTIINTENT_AUG_CSV
_orig_path = opt.MULTIINTENT_CSV

# Use augmented only if toggled ON AND the file exists
if USE_AUGMENTED and os.path.exists(_aug_path):
    _dataset_path = _aug_path
else:
    _dataset_path = _orig_path
dsqi = pd.read_csv(_dataset_path, encoding='utf-8-sig') # dsqi = dataset question intent

tokenizer = AutoTokenizer.from_pretrained(opt.MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer.to(device)

# Pecah string intent yang dibatasi titik koma (;) menjadi sebuah List
dsqi['intent_list'] = dsqi['Intent'].apply(lambda x: x.split('; '))

# Terapkan Multi-Hot Encoding menggunakan mlb
mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(dsqi['intent_list'])
listof_intent = list(mlb.classes_)

# Tokenisasi inten
tokenized_intent = tokenizer(
    listof_intent,
    padding = True,
    truncation = True,
    return_tensors = 'pt'
)

# Hasil encoding menjadi df baru dan diubah menjadi float32
df_inten = pd.DataFrame(encoded_labels, columns=mlb.classes_).astype('float32')

# Gabungkan df_inten dengan dsqi
dsqi_encoded = pd.concat([dsqi['question'], df_inten], axis=1)

class LABANDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=opt.max_len):
        self.data = dataframe 
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_columns = self.data.columns[1:]  # Kolom label dimulai dari indeks 1

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Ambil baris data berdasarkan indeks
        row = self.data.iloc[idx]
        question = row['question']

        # ambil nilai label untuk setiap kelas intent
        label_array = row[self.label_columns].values.astype('float32')  # Pastikan label dalam format float

        # tokenisasi question
        encoding = self.tokenizer(
            question,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids' : encoding['input_ids'].flatten(),
            'attention_mask' : encoding['attention_mask'].flatten(),
            'labels' : torch.FloatTensor(label_array)
        }    

# inisialisasi dataset untuk train 
dataset_train = LABANDataset(
    dataframe = dsqi_encoded,
    tokenizer = tokenizer
)

train_dataload = DataLoader(
    dataset_train,
    batch_size=16,
    shuffle=True
)

batch = next(iter(train_dataload))
print("Input IDs (question):", batch['input_ids'].shape)
print("Labels (Kunci intent):", batch['labels'].shape)

# membelah data menjadi 80% training, 10% validasi, 10% test
df_train, df_temp = train_test_split(dsqi_encoded, test_size=0.2, random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

# define 3 dataset
train_data = LABANDataset(dataframe=df_train, tokenizer=tokenizer)
val_data = LABANDataset(dataframe=df_val, tokenizer=tokenizer)
test_data = LABANDataset(dataframe=df_test, tokenizer=tokenizer)

# membuat tiga antrian data untuk train, validasi, dan test
train_dataload = DataLoader(
    train_data, batch_size=opt.BATCH_SIZE, shuffle=True
)
val_dataload = DataLoader(
    val_data, batch_size=opt.BATCH_SIZE, shuffle=False
)
test_dataload = DataLoader(
    test_data, batch_size=opt.BATCH_SIZE, shuffle=False
)
