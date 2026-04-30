import json
import random
from core.vector_db import VectorDBManager

class KnowledgeBaseEngine:
    def __init__(self):
        self.vector_db = VectorDBManager()
        # Ensure indices are built
        self.vector_db.build_ayat_index('data/dataset_ayat.csv')
        self.vector_db.build_qna_index('data/dataset_qna.csv')
        
        # Mapping for simple stages
        self.stages = ['pembukaan', 'pembahasan', 'intervensi', 'solusi', 'relaksasi', 'penutupan']
        
        # Load answer templates from answer.json
        try:
            with open('data/answer.json', 'r', encoding='utf-8') as f:
                self.answers = json.load(f)
        except Exception as e:
            print(f"Failed to load answer.json: {e}")
            self.answers = {"tahapan": []}
            
    def get_template_answer(self, stage):
        """Ambil jawaban template berdasarkan tahapan"""
        for tahapan in self.answers.get('tahapan', []):
            if tahapan.get('tag') == stage:
                answers = tahapan.get('answer', [])
                # Filter out empty answers
                valid_answers = [ans for ans in answers if ans.strip()]
                if valid_answers:
                    return random.choice(valid_answers)
        return ""

    def process_turn(self, user_text, detected_intents, current_stage):
        """
        Memproses giliran percakapan.
        Mengembalikan response chatbot dan state (stage) berikutnya.
        """
        response = ""
        next_stage = current_stage
        
        # 1. Jika stage = pembukaan atau penutupan
        if current_stage in ['pembukaan', 'penutupan']:
            template_ans = self.get_template_answer(current_stage)
            if template_ans:
                response = template_ans
            else:
                if current_stage == 'pembukaan':
                    response = "Halo, selamat datang di sesi konseling ini. Bagaimana perasaanmu hari ini?"
                else:
                    response = "Terima kasih telah berbagi hari ini. Semoga harimu menyenangkan dan Tuhan memberkati."
            
            # Pindah ke pembahasan setelah pembukaan
            if current_stage == 'pembukaan':
                next_stage = 'pembahasan'
                return response, next_stage
            else:
                return response, current_stage
                
        # 2. Jika stage = pembahasan
        elif current_stage == 'pembahasan':
            # Cari dari dataset_qna via FAISS jika ada yang mirip
            qna_results = self.vector_db.search_qna(user_text, k=1)
            
            if qna_results:
                response = qna_results[0].metadata.get('answer', '')
            
            if not response:
                if detected_intents:
                    intents_str = ", ".join(detected_intents)
                    response = f"Saya memahami bahwa kamu sedang merasakan {intents_str}. Bisa ceritakan lebih lanjut tentang hal itu?"
                else:
                    response = "Saya mengerti. Bisa kamu ceritakan lebih detail lagi tentang apa yang kamu rasakan?"
            
            # Tanya konfirmasi untuk pindah stage
            response += "\n\nApakah masih ada yang ingin kamu ceritakan atau kita bisa lanjut membahas langkah selanjutnya?"
            # State tetap di pembahasan, user akan merespon iya/tidak
            
        # 3. Transisi dari pembahasan ke intervensi/solusi
        # Di sini kita asumsikan state machine dikendalikan di luar kelas ini juga, 
        # namun jika user memicu intervensi:
        elif current_stage in ['intervensi', 'solusi']:
            response = "Mari kita coba melihat dari sudut pandang yang berbeda. "
            
            if detected_intents:
                primary_intent = detected_intents[0]
                ayat_results = self.vector_db.get_ayat_by_intent(primary_intent, k=1)
                
                if ayat_results:
                    ayat = ayat_results[0].metadata.get('ayat', '')
                    response += f"\nMengingat perasaanmu tentang {primary_intent}, ingatlah firman Tuhan ini:\n{ayat}\n"
            
            response += "\nBagaimana menurutmu tentang ayat ini dan apakah kamu merasa lebih tenang?"
            next_stage = 'relaksasi'
            
        elif current_stage == 'relaksasi':
            response = "Mari kita ambil waktu sejenak untuk menarik napas dalam-dalam. Fokus pada ketenangan dan biarkan bebanmu sedikit terangkat."
            next_stage = 'penutupan'
            
        return response, next_stage

if __name__ == "__main__":
    engine = KnowledgeBaseEngine()
    resp, stage = engine.process_turn("Saya sangat sedih dan merasa kehilangan arah", ["Perasaan Sedih dan Kehilangan"], "intervensi")
    print(resp)
    print("Next stage:", stage)
