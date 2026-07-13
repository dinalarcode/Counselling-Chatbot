from core.rag_engine.rag_engine import RAGEngine
from core.session_manager.SM_Engine import SessionManager
import logging

# Set up logging to only show critical errors (suppresses transformers/langchain warnings)
logging.getLogger("transformers").setLevel(logging.ERROR)

def run_test():
    print("Inisialisasi RAG Engine (Loading Model & VectorDB)...")
    try:
        engine = RAGEngine()
        session = SessionManager(engine)
        print("Berhasil! RAG Engine & Session Manager siap digunakan.\n")
    except ValueError as e:
        print(f"Error: {e}")
        print("Pastikan Anda sudah membuat file .env dan mengisi GEMINI_API_KEY.")
        return
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat model: {e}")
        return

    print("=" * 60)
    print("  SESI KONSELING INTERAKTIF (Automatic Stage Progression)")
    print("=" * 60)
    print("Perintah khusus:")
    print("  'done'   → Keluar dari sesi")
    print("  'next'   → Paksa lanjut ke tahap berikutnya (dev only)")
    print("  'state'  → Lihat state sesi saat ini (dev only)")
    print("=" * 60)
    print()

    while True:
        state = session.get_state()
        stage_label = state['stage'].upper()
        turn = state['turn_count']
        print(f"--- [Tahap: {stage_label} | Giliran ke-{turn}] ---")
        pasien_input = input("Pasien : ")

        if pasien_input.strip().lower() == 'done':
            print("\nSesi konseling diakhiri. Semoga harimu menyenangkan!")
            break

        if pasien_input.strip().lower() == 'next':
            new_stage = session.force_advance()
            if new_stage:
                print(f"[DEV] Paksa beralih ke tahap: {new_stage.upper()}\n")
            else:
                print("[DEV] Sudah di tahap terakhir.\n")
            continue

        if pasien_input.strip().lower() == 'state':
            print(f"\n[DEV STATE] {state}\n")
            continue

        print("Memproses...\n")

        try:
            result = session.chat(pasien_input)
            debug = result['debug']

            # Print debug context
            print("\n[INFO KONTEKS]")
            print(f"  Tahap          : {debug['stage']}")
            print(f"  Giliran        : {debug['turn_count']}")
            print(f"  Intent (turn)  : {debug['intents_this_turn']}")
            print(f"  Intent (akum.) : {debug['primary_intents']}")
            print(f"  Ayat Alkitab   : {debug['bible_verses'][:80]}..." if debug['bible_verses'] else "  Ayat Alkitab   : -")

            if debug['transitioned']:
                print(f"  >>> TRANSISI    : {debug['stage']} -> {debug['new_stage']}")

            # Print the response
            print(f"\nPsikolog:\n{result['response']}\n")

        except Exception as e:
            print(f"Error saat menghasilkan respons: {e}\n")

if __name__ == "__main__":
    run_test()
