"""RAG engine orchestrating intent classification, FAISS retrieval, LLM reranking, and response generation."""

import os
import time
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# Load the predictor and VectorDBManager.
from models.multilabel.predict import predictor
from core.vectordb_manager.vectordb_engine import VectorDBManager
from config import opt

# Static prompts and config lists for the engine.
from core.rag_engine.Listof_RAGlist import (
    STAGE_INSTRUCTIONS, PHYSICAL_SYMPTOM_CBT_GUIDANCE, SPIRITUAL_CONSENT_PROMPT,
    BIBLE_VERSE_STAGES, PEMBAHASAN_TURN3_PROMPT, SKIP_CLASSIFICATION_STAGES,
    MAIN_PROMPT_TEMPLATE,
)

# Load environment variables.
load_dotenv()


def _build_chatbot_llm():
    """Build the LLM instance based on opt.CHATBOT_LLM_PROVIDER."""
    provider = opt.CHATBOT_LLM_PROVIDER.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv(opt.GROQ_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"{opt.GROQ_API_KEY_ENV} not found in .env")
        return ChatGroq(
            model=opt.GROQ_CHATBOT_MODEL,
            api_key=api_key,
            temperature=opt.CHATBOT_TEMPERATURE,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv(opt.GEMINI_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"{opt.GEMINI_API_KEY_ENV} not found in .env")
        return ChatGoogleGenerativeAI(
            model=opt.GEMINI_CHATBOT_MODEL,
            google_api_key=api_key,
            temperature=opt.CHATBOT_TEMPERATURE,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv(opt.OPENAI_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"{opt.OPENAI_API_KEY_ENV} not found in .env")
        return ChatOpenAI(
            model=opt.OPENAI_CHATBOT_MODEL,
            api_key=api_key,
            temperature=opt.CHATBOT_TEMPERATURE,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model="qwen2.5:3b",
            base_url="http://localhost:11434",
            temperature=opt.CHATBOT_TEMPERATURE,
        )

    else:
        raise ValueError(
            f"Unknown CHATBOT_LLM_PROVIDER: '{provider}'. "
            f"Pilihan valid: 'groq', 'gemini', 'openai', 'ollama'"
        )

class RAGEngine:
    # Static prompts and config imported from Listof_RAGlist and bound as class attributes so self access keeps working.
    STAGE_INSTRUCTIONS = STAGE_INSTRUCTIONS
    PHYSICAL_SYMPTOM_CBT_GUIDANCE = PHYSICAL_SYMPTOM_CBT_GUIDANCE
    SPIRITUAL_CONSENT_PROMPT = SPIRITUAL_CONSENT_PROMPT
    BIBLE_VERSE_STAGES = BIBLE_VERSE_STAGES
    PEMBAHASAN_TURN3_PROMPT = PEMBAHASAN_TURN3_PROMPT
    SKIP_CLASSIFICATION_STAGES = SKIP_CLASSIFICATION_STAGES

    def __init__(self):
        # Initialize the LLM with the provider set in config.py.
        self.llm = _build_chatbot_llm()

        # Initialize the multilabel classifier predictor.
        self.classifier = predictor()

        # Initialize the vector DB manager knowledge base.
        self.vector_db = VectorDBManager()
        # Build indices if not already built.
        self.vector_db.build_bible_index('data/verse_retrieval/alkitab_tb_enriched_groq.csv')
        self.vector_db.build_qna_index('data/dataset_qna.csv')

        # Define the prompt template where stage_instruction replaces the stage label and bible_section is conditional.
        self.prompt_template = PromptTemplate(
            input_variables=[
                "user_input",
                "detected_intents",
                "stage_instruction",
                "example_answer",
                "bible_section"
            ],
            template=MAIN_PROMPT_TEMPLATE
        )

        # Create the LangChain processing chain.
        self.chain = self.prompt_template | self.llm

    def generate_background_summary(self, history_list):
        """Summarize the pembahasan history into one sentence at the pembahasan to intervensi transition."""
        if not history_list:
            return ""

        conversation_text = ""
        for i, (user_msg, bot_resp) in enumerate(history_list, 1):
            conversation_text += f"[Giliran {i}]\nKlien: {user_msg}\nKonselor: {bot_resp}\n\n"

        prompt = (
            "Berikut adalah riwayat percakapan antara klien dan konselor pada tahap eksplorasi masalah:\n\n"
            f"{conversation_text}"
            "Buat SATU kalimat ringkasan yang menangkap inti keluhan utama klien. "
            "Fokus pada perasaan dan situasi klien, bukan pada respons konselor. "
            "Gunakan Bahasa Indonesia. Jawab hanya dengan satu kalimat ringkasan, tanpa penjelasan tambahan."
        )

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            summary = response.content.strip()
            print(f"[Summary] Background summary generated: {summary}")
            return summary
        except Exception as e:
            print(f"[Summary] Failed to generate background summary: {e}")
            return ""

    def generate_technique_extraction(self, history_list):
        """Extract the relaxation technique chosen during solusi at the solusi to relaksasi transition."""
        if not history_list:
            return None

        conversation_text = ""
        for i, (user_msg, bot_resp) in enumerate(history_list, 1):
            conversation_text += f"[Giliran {i}]\nKlien: {user_msg}\nKonselor: {bot_resp}\n\n"

        prompt = (
            "Berikut adalah riwayat percakapan antara klien dan konselor pada tahap pemberian solusi:\n\n"
            f"{conversation_text}"
            "Berdasarkan riwayat singkat ini, teknik relaksasi apa yang dipilih atau disepakati untuk klien? "
            "Jawab singkat nama tekniknya saja (contoh: 'Pernapasan 4-7-8' atau 'Grounding 5-4-3-2-1'). "
            "Jika tidak ada teknik yang disebutkan secara eksplisit, jawab dengan tepat: 'tidak ada'."
        )

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            technique = response.content.strip()
            if not technique or technique.lower() == 'tidak ada':
                return None
            print(f"[Technique] Chosen technique extracted: {technique}")
            return technique
        except Exception as e:
            print(f"[Technique] Failed to extract chosen technique: {e}")
            return None

    def generate_response(self, user_input, current_stage="pembahasan", override_intents=None, has_physical_symptoms=False, excluded_books=None, excluded_verses=None, spiritual_consent=None, ask_spiritual_consent=False, turn_in_stage=0, complaint_summary=None, chosen_technique=None):
        """Generate a counseling response for the given stage, intents, consent, and session context."""
        # Get the stage-specific behavioral instruction.
        stage_instruction = self.STAGE_INSTRUCTIONS.get(
            current_stage,
            self.STAGE_INSTRUCTIONS["pembahasan"]  # Fallback instruction.
        )

        # Inject the ephemeral complaint summary into later stages so the LLM retains the core issue across turns.
        if complaint_summary and current_stage in ('intervensi', 'solusi', 'relaksasi'):
            stage_instruction = f"Konteks Keluhan Klien: {complaint_summary}\n\n" + stage_instruction

        # Inject the chosen relaxation technique for relaksasi so the LLM only guides the agreed technique.
        if chosen_technique and current_stage == 'relaksasi':
            stage_instruction = (
                f"Klien telah memilih teknik {chosen_technique}. "
                f"Pandu klien HANYA dengan teknik tersebut. "
            ) + stage_instruction

        # After 3 pembahasan turns the LLM must explicitly ask if the user has anything else to share.
        if current_stage == 'pembahasan' and turn_in_stage >= 3:
            stage_instruction = stage_instruction + " " + self.PEMBAHASAN_TURN3_PROMPT

        # Inject CBT physical symptom guidance when applicable.
        if has_physical_symptoms and current_stage in ('solusi', 'relaksasi'):
            stage_instruction = stage_instruction + " " + self.PHYSICAL_SYMPTOM_CBT_GUIDANCE

        # Ask for spiritual consent at the final solusi turn so consent resolves before any verse is introduced.
        if ask_spiritual_consent and current_stage == 'solusi':
            stage_instruction = stage_instruction + " " + self.SPIRITUAL_CONSENT_PROMPT

        _t0 = time.perf_counter()

        # Intent classification is skipped in stages where it has no effect on the response.
        if current_stage in self.SKIP_CLASSIFICATION_STAGES:
            detected_intents = []
            intents_str = "Umum"
            _t1 = time.perf_counter()
            print(f"[TIMING] classifier.predict: skipped (stage={current_stage})")
        else:
            prediction = self.classifier.predict(user_input)
            detected_intents = prediction.get('inten_terdeteksi', [])
            intents_str = ", ".join(detected_intents) if detected_intents else "Umum"
            _t1 = time.perf_counter()
            print(f"[TIMING] classifier.predict: {(_t1 - _t0) * 1000:.1f}ms")

        # Get the example answer from the QnA DB.
        qna_results = self.vector_db.search_qna(user_input, k=1)
        example_answer = ""
        if qna_results:
            example_answer = qna_results[0].metadata.get('answer', '')
        _t2 = time.perf_counter()
        print(f"[TIMING] search_qna: {(_t2 - _t1) * 1000:.1f}ms")

        # Retrieve bible verses only during the verse stages using the LLM reranker over FAISS top-10 candidates.
        bible_verses = ""
        bible_reference = ""
        bible_book_abbr = ""
        # Conditional RAG where verses are retrieved only when the user has explicitly given spiritual consent.
        intents_for_verse = override_intents if override_intents else detected_intents
        if current_stage in self.BIBLE_VERSE_STAGES and intents_for_verse and spiritual_consent is True:
            verse_results = self.vector_db.retrieve_verse_with_llm(
                intents=intents_for_verse,
                user_input=user_input,
                llm=self.llm,
                k=10,
                excluded_books=excluded_books,
                excluded_verses=excluded_verses
            )
            if verse_results:
                bible_reference = verse_results[0].get('reference', '')
                bible_verses = verse_results[0].get('text', '')
                bible_book_abbr = verse_results[0].get('book_abbr', '')
        _t3 = time.perf_counter()
        if current_stage in self.BIBLE_VERSE_STAGES:
            print(f"[TIMING] retrieve_verse_with_llm: {(_t3 - _t2) * 1000:.1f}ms")

        # Build the bible section dynamically and only include it when a verse is present.
        bible_section = ""
        if bible_verses:
            verse_display = f"{bible_reference} (TB) \"{bible_verses}\""
            bible_section = (
                "- Ayat Alkitab Relevan (Sertakan dan jelaskan maknanya dengan lembut dalam konteks masalah pengguna):\n"
                f'{verse_display}'
            )

        # Format the inputs and invoke the LLM chain.
        response = self.chain.invoke({
            "user_input": user_input,
            "detected_intents": intents_str,
            "stage_instruction": stage_instruction,
            "example_answer": example_answer,
            "bible_section": bible_section
        })

        _t4 = time.perf_counter()
        print(f"[TIMING] chain.invoke (LLM): {(_t4 - _t3) * 1000:.1f}ms")
        print(f"[TIMING] total generate_response: {(_t4 - _t0) * 1000:.1f}ms")

        # Return the generated text and the context used for development and debugging.
        return {
            "response": response.content,
            "context_used": {
                "intents": detected_intents,
                "example_answer": example_answer,
                "bible_verses": f"{bible_reference} - {bible_verses}" if bible_verses else "",
                "bible_reference": bible_reference,
                "bible_book_abbr": bible_book_abbr,
                "stage": current_stage
            }
        }
