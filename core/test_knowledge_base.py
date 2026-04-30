from core.knowledge_base import KnowledgeBaseEngine

def run_test_session():
    engine = KnowledgeBaseEngine()
    print("\n--- Memulai Simulasi Sesi Konseling ---")
    
    # 1. Pembukaan
    stage = 'pembukaan'
    user_input = ""
    print(f"\n[Stage: {stage}]")
    resp, stage = engine.process_turn(user_input, [], stage)
    print(f"Chatbot: {resp}")
    
    # 2. Pembahasan
    print(f"\n[Stage: {stage}]")
    user_input = "Saya merasa sangat lelah dan tidak berharga belakangan ini."
    intents = ["Perasaan Sedih dan Kehilangan"]
    print(f"User: {user_input} (Detected: {intents})")
    
    resp, stage = engine.process_turn(user_input, intents, stage)
    print(f"Chatbot: {resp}")
    
    # Simulasi user menjawab konfirmasi dengan "Lanjut" yang mengubah stage ke intervensi
    print("\n*User setuju untuk lanjut ke tahap berikutnya*")
    stage = 'intervensi'
    
    # 3. Intervensi
    print(f"\n[Stage: {stage}]")
    user_input = "Iya, mari kita bahas solusinya."
    print(f"User: {user_input}")
    
    resp, stage = engine.process_turn(user_input, intents, stage)
    print(f"Chatbot: {resp}")
    
    # 4. Relaksasi
    print(f"\n[Stage: {stage}]")
    user_input = "Terima kasih, ayat itu sangat membantu."
    print(f"User: {user_input}")
    
    resp, stage = engine.process_turn(user_input, [], stage)
    print(f"Chatbot: {resp}")
    
    # 5. Penutupan
    print(f"\n[Stage: {stage}]")
    user_input = "Sesi ini sangat berarti buat saya."
    print(f"User: {user_input}")
    
    resp, stage = engine.process_turn(user_input, [], stage)
    print(f"Chatbot: {resp}")
    
    print("\n--- Sesi Selesai ---")

if __name__ == "__main__":
    run_test_session()
