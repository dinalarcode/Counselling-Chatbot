import pandas as pd
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorDBManager:
    def __init__(self, model_name='indobenchmark/indobert-base-p1'):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.ayat_db = None
        self.qna_db = None

    def build_ayat_index(self, csv_path='data/dataset_ayat.csv'):
        # Membaca dataset ayat
        df_ayat = pd.read_csv(csv_path)
        
        documents = []
        for index, row in df_ayat.iterrows():
            intent = str(row['Inten']).strip()
            ayat = str(row['Ayat Alkitab']).strip()
            
            # The text we embed is the intent so we can search by intent.
            # But wait, FAISS search will compare query embedding with document embedding.
            # Usually, the user query will be embedded. We want to find the verse that matches
            # the query's meaning or the detected intent. If we search by intent, we can embed the intent name.
            # To be richer, we embed both intent and the verse itself, but mainly the intent for counseling context.
            page_content = f"Intent: {intent} | Ayat: {ayat}"
            doc = Document(page_content=page_content, metadata={"intent": intent, "ayat": ayat})
            documents.append(doc)

        self.ayat_db = FAISS.from_documents(documents, self.embeddings)
        print("Ayat index built successfully.")

    def get_ayat_by_intent(self, intent_name, k=1):
        if self.ayat_db is None:
            self.build_ayat_index()
            
        # We can search the FAISS index
        # To make it accurate for a specific intent, we just search the intent string
        results = self.ayat_db.similarity_search(intent_name, k=k)
        return results

    def build_qna_index(self, qna_csv_path='data/dataset_qna.csv'):
        df_qna = pd.read_csv(qna_csv_path)
        
        documents = []
        # Asumsi kolom question dan answer
        for index, row in df_qna.iterrows():
            question = str(row['question']).strip()
            answer = str(row['answer']).strip()
            if pd.isna(question) or question == 'nan':
                continue
                
            page_content = f"Pertanyaan: {question}"
            doc = Document(page_content=page_content, metadata={"answer": answer})
            documents.append(doc)
            
        if documents:
            self.qna_db = FAISS.from_documents(documents, self.embeddings)
            print("QnA index built successfully.")
        else:
            print("No valid QnA documents found.")

    def search_qna(self, query, k=1):
        if self.qna_db is None:
            self.build_qna_index()
        results = self.qna_db.similarity_search(query, k=k)
        return results

if __name__ == "__main__":
    db_manager = VectorDBManager()
    db_manager.build_ayat_index('../data/dataset_ayat.csv')
    db_manager.build_qna_index('../data/dataset_qna.csv')
    
    # Test search
    res = db_manager.get_ayat_by_intent("Perasaan Sedih dan Kehilangan")
    for r in res:
        print(r.metadata['ayat'])
