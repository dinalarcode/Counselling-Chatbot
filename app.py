import os
import sys
from flask import Flask, render_template, request, jsonify

# Add project root to sys.path to allow importing from core
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.rag_engine import RAGEngine
from core.session_manager import SessionManager

app = Flask(__name__)

# Initialize global instances for the chatbot
print("Initializing RAGEngine and SessionManager...")
try:
    rag = RAGEngine()
    chatbot_session = SessionManager(rag)
    print("Initialization complete.")
except Exception as e:
    print(f"Error during initialization: {e}")
    rag = None
    chatbot_session = None

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

    user_input = data["message"]
    
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
        
        return jsonify({"response": result["response"]})
    
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
