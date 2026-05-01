from core.rag_engine import RAGEngine
import logging

# Set up logging to only show critical errors (suppresses transformers/langchain warnings)
logging.getLogger("transformers").setLevel(logging.ERROR)

def run_test():
    print("Inisialisasi RAG Engine (Loading Model & VectorDB)...")
    try:
        engine = RAGEngine()
        print("Berhasil! RAG Engine siap digunakan.\n")
    except ValueError as e:
        print(f"Error: {e}")
        print("Pastikan Anda sudah membuat file .env dan mengisi GEMINI_API_KEY.")
        return
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat model: {e}")
        return

    # List tahapan konseling
    stages = ['pembukaan', 'pembahasan', 'intervensi', 'solusi', 'relaksasi', 'penutupan']
    current_stage_index = 1 # Mulai dari pembahasan

    print("Ketik 'done' untuk keluar.")
    print("Ketik 'next' untuk lanjut ke tahap konseling berikutnya.\n")
    
    while True:
        current_stage = stages[current_stage_index]
        print(f"--- [Tahap Saat Ini: {current_stage.upper()}] ---")
        pasien_input = input("Pasien : ")
        
        if pasien_input.lower() == 'done':
            print("Sesi konseling diakhiri.")
            break
            
        if pasien_input.lower() == 'next':
            current_stage_index = min(current_stage_index + 1, len(stages) - 1)
            print(f"Beralih ke tahap: {stages[current_stage_index].upper()}\n")
            continue

        print("Memproses (Menganalisis Niat, Mencari di Knowledge Base, dan Menghasilkan Respons)...\n")
        
        try:
            result = engine.generate_response(pasien_input, current_stage=current_stage)
            
            # Print Context Used for debugging/visibility
            context = result['context_used']
            print("\n[INFO KONTEKS]")
            print(f"- Niat (Intent)  : {context['intents']}")
            print(f"- Ayat Alkitab   : {context['bible_verses']}")
            print(f"- Referensi QnA  : {context['example_answer'][:100]}...\n")
            
            # Print Final LLM Response
            print(f"Psikolog (Gemini): \n{result['response']}\n")
        except Exception as e:
            print(f"Error saat menghasilkan respons: {e}\n")

if __name__ == "__main__":
    run_test()
