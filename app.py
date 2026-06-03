import os
import sys
from flask import Flask, render_template, request, jsonify

# Add project root to sys.path to allow importing from core
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.rag_engine import RAGEngine
from core.session_manager import SessionManager


# ── Bible Test Session (sandbox for verse retrieval testing) ────────────

class BibleTestSession:
    """
    Sandbox session for testing FAISS Bible verse retrieval.
    Bypasses the LLM entirely — only runs LABAN classifier + FAISS.
    Used by respondents to evaluate verse relevance (Cohen's Kappa).
    """

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.active = False

    def handle(self, user_input: str) -> str:
        """
        Run LABAN classification + FAISS retrieval + LLM reranking.
        Returns a formatted string with only the verse — no counseling LLM.
        """
        # 1. Classify intents via LABAN
        prediction = self.rag_engine.classifier.predict(user_input)
        detected_intents = prediction.get('inten_terdeteksi', [])
        scores = prediction.get('skore', {})

        # 2. Retrieve verse via FAISS + LLM Reranker
        verse_results = self.rag_engine.vector_db.retrieve_verse_with_llm(
            intents=detected_intents,
            user_input=user_input,
            llm=self.rag_engine.llm,
            k=5
        )

        # 3. Format output
        intents_str = ", ".join(detected_intents) if detected_intents else "(tidak ada intent terdeteksi)"

        if not verse_results:
            return (
                f"**Intent Terdeteksi:** {intents_str}\n\n"
                f"⚠️ Tidak ada ayat yang ditemukan untuk input ini."
            )

        verse = verse_results[0]

        # Console log for developer to record into Cohen's Kappa sheet
        print(f"\n--- Bible Test (FAISS + LLM Reranker) ---")
        print(f"Utterance: {user_input}")
        print(f"Intents:   {intents_str}")
        print(f"Verse:     {verse['reference']} — {verse['text']}")
        print(f"------------------------------------------\n")

        return (
            f"**Intent Terdeteksi:** {intents_str}\n\n"
            f"**Ayat:** {verse['reference']} (TB)\n\n"
            f"\"{ verse['text']}\"\n\n"
            f"*(dipilih oleh LLM reranker dari 5 kandidat FAISS)*"
        )


# ── Flask app ───────────────────────────────────────────────────────────

app = Flask(__name__)

# Initialize global instances for the chatbot
print("Initializing RAGEngine and SessionManager...")
try:
    rag = RAGEngine()
    chatbot_session = SessionManager(rag)
    bible_test_session = BibleTestSession(rag)
    print("Initialization complete.")
except Exception as e:
    print(f"Error during initialization: {e}")
    rag = None
    chatbot_session = None
    bible_test_session = None

@app.route("/")
def index():
    """Render the main chat interface."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Receive user message and return chatbot response."""
    if not chatbot_session:
        return jsonify({"error": "Chatbot is not initialized correctly. Check terminal logs."}), 500

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided."}), 400

    user_input = data["message"].strip()

    # ── Bible Test Mode gate ─────────────────────────────────────────
    if user_input.lower() == "bible test" and bible_test_session:
        bible_test_session.active = True
        print("\n--- Bible Test Mode ACTIVATED ---\n")
        return jsonify({"response": (
            "🔬 **Mode Bible Test Aktif**\n\n"
            "Masukkan utterance dari daftar pengujian Anda.\n"
            "Pipeline pengambilan ayat:\n"
            "1. Klasifikasi intent (LABAN/IndoBERT)\n"
            "2. FAISS semantic search → 5 kandidat ayat\n"
            "3. LLM reranker (Gemini) → memilih ayat terbaik\n\n"
            "Ketik `exit test` untuk kembali ke sesi normal."
        )})

    if user_input.lower() == "exit test" and bible_test_session:
        bible_test_session.active = False
        print("\n--- Bible Test Mode DEACTIVATED ---\n")
        return jsonify({"response": "✅ Mode Bible Test dinonaktifkan. Sesi konseling normal dilanjutkan."})

    if bible_test_session and bible_test_session.active:
        try:
            verse_response = bible_test_session.handle(user_input)
            return jsonify({"response": verse_response})
        except Exception as e:
            print(f"Error during bible test: {e}")
            return jsonify({"error": "Terjadi kesalahan saat mengambil ayat."}), 500
    # ─────────────────────────────────────────────────────────────────

    # Process the message through the session manager
    try:
        result = chatbot_session.chat(user_input)
        
        # We only return the response to the user.
        # Technical details (debug data) are printed to the console.
        print("\n--- Technical Debug Info ---")
        print(f"User Input: {user_input}")
        print(f"Stage: {result['debug']['stage']} -> {result['debug']['new_stage']}")
        print(f"Intents: {result['debug']['intents_this_turn']}")
        if result['debug'].get('bible_verses'):
            print(f"Bible Verses Used: {result['debug']['bible_verses']}")
        print("----------------------------\n")
        
        return jsonify({
            "response": result["response"],
            "show_professional_button": result.get("show_professional_button", False)
        })
    
    except Exception as e:
        print(f"Error during chat processing: {e}")
        return jsonify({"error": "Terjadi kesalahan internal. Silakan coba lagi."}), 500

@app.route("/reset", methods=["POST"])
def reset():
    """Reset the chatbot session."""
    if chatbot_session:
        chatbot_session.reset()
        print("\n--- Session Reset ---")
    return jsonify({"status": "success", "message": "Session reset."})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
