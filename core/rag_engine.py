import os
import time
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# Load the predictor and VectorDBManager
from models.multilabel.predict import predictor
from core.vector_db import VectorDBManager
from config import opt

# Load environment variables
load_dotenv()


def _build_chatbot_llm():
    """Build LLM instance based on opt.CHATBOT_LLM_PROVIDER."""
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
    # Stage-to-behavioral-instruction mapping
    # These guide Gemini's tone and approach without revealing the stage label to the user
    STAGE_INSTRUCTIONS = {
        "pembukaan": (
            "Anda sedang memulai percakapan pertama dengan pengguna. "
            "Sambutlah dengan hangat, perkenalkan diri sebagai konselor yang siap mendengarkan, "
            "dan tanyakan bagaimana perasaan mereka hari ini. "
            "Jangan memberikan nasihat atau ayat Alkitab di tahap ini. "
            "Buat pengguna merasa aman dan nyaman untuk bercerita."
            "Jangan bertele-tele dalam menanggapi."
        ),
        "pembahasan": (
            "Pengguna sedang menceritakan masalahnya. "
            "DILARANG KERAS memberikan solusi, saran, nasihat, atau reframing positif. "
            "Tugas Anda SAAT INI HANYA mengeksplorasi masalah dan memvalidasi emosi klien "
            "menggunakan pertanyaan terbuka. "
            "Dengarkan dengan empati, akui dan validasi perasaan mereka tanpa menghakimi. "
            "Jika masih kurang jelas, ajukan pertanyaan terbuka untuk memahami situasi lebih dalam. "
            "Namun, jika pengguna sudah menceritakan inti masalahnya atau terlihat sudah selesai, "
            "jangan terus memaksa mereka bercerita. "
            "Jangan menyertakan ayat Alkitab di tahap ini. "
            "Jangan bertele-tele dalam menanggapi."
        ),
        "intervensi": (
            "Anda masih berada dalam tahap memahami pengguna secara lebih mendalam. "
            "DILARANG KERAS memberikan solusi, saran, nasihat, atau reframing positif. "
            "Tugas Anda SAAT INI HANYA mengeksplorasi masalah lebih dalam dan memvalidasi emosi "
            "klien menggunakan pertanyaan terbuka. "
            "Gali akar perasaan dan pikiran pengguna dengan lembut, pastikan mereka merasa "
            "benar-benar didengar dan dipahami sebelum melangkah lebih jauh. "
            "Jangan menyertakan ayat Alkitab di tahap ini. "
            "Jangan bertele-tele dalam menanggapi."
        ),
        "solusi": (
            "Bantu pengguna merumuskan langkah-langkah konkret yang bisa mereka ambil. "
            "Berikan saran praktis yang realistis dan dorong mereka untuk bertindak. "
            "Jangan menyertakan ayat Alkitab di tahap ini. "
            "Berikan jawaban yang singkat, padat, dan bermakna."
        ),
        "relaksasi": (
            "Bantu pengguna untuk benar-benar menenangkan diri setelah diskusi yang panjang. "
            "WAJIB pandu pengguna MELAKUKAN teknik relaksasi psikologis yang konkret secara langkah demi langkah. "
            "Jika teknik spesifik klien tidak disebutkan di atas, pilih salah satu: "
            "(a) Teknik Pernapasan 4-7-8 — tarik napas 4 detik, tahan 7 detik, hembuskan 8 detik, ulangi; atau "
            "(b) Teknik Grounding 5-4-3-2-1 — 5 hal dilihat, 4 didengar, 3 disentuh, 2 dicium, 1 dirasakan. "
            "Bimbing dengan nada yang menenangkan, penuh kasih, dan perlahan. "
            "Jika tersedia 'Ayat Alkitab Relevan', integrasikan secara halus ke dalam latihan "
            "sebagai penguat ketenangan — bukan ceramah. "
            "Jika TIDAK tersedia ayat, fokus pada teknik dan JANGAN menambahkan ayat sendiri. "
            "PENTING — Aturan Anti-Looping: Jika ucapan klien mengisyaratkan mereka sudah selesai "
            "melakukan teknik (contoh: 'udah', 'selesai', 'sudah mendingan', 'lebih baik', 'lega'), "
            "DILARANG mengulangi instruksi teknik. "
            "Anda HARUS bertanya: 'Bagaimana perasaanmu sekarang setelah melakukannya?'"
        ),
        "penutupan": (
            "Sesi konseling hampir selesai. Ringkas poin-poin penting yang sudah dibahas, "
            "berikan dorongan semangat terakhir, dan ucapkan terima kasih atas keterbukaan pengguna. "
            "Akhiri dengan doa singkat atau harapan baik. "
            "Jangan menyertakan ayat Alkitab baru, cukup berikan penutupan yang hangat. "
            "Di akhir respons, tawarkan dengan lembut: "
            "'Jika kamu merasa membutuhkan pendampingan lebih lanjut dari konselor profesional, "
            "kami bisa membantu menghubungkanmu. Apakah kamu mau?'"
        ),
        "bantuan_profesional": (
            "Pengguna menunjukkan tanda-tanda bahwa mereka membutuhkan bantuan dari konselor profesional. "
            "Sampaikan dengan penuh empati bahwa keputusan mereka untuk mencari bantuan adalah langkah yang sangat berani dan tepat. "
            "Tegaskan bahwa mencari pertolongan bukanlah tanda kelemahan, melainkan tanda kekuatan dan keberanian. "
            "Jika tersedia 'Ayat Alkitab Relevan', gunakan secara lembut untuk menguatkan bahwa Tuhan mendukung mereka. "
            "Informasikan bahwa mereka dapat melanjutkan untuk berbicara dengan konselor profesional "
            "dengan menekan tombol yang akan muncul di layar. "
            "Gunakan nada yang menenangkan, penuh kasih, dan memberikan harapan."
        ),
        "penutupan_selesai": (
            "Pengguna telah memilih untuk mengakhiri sesi konseling tanpa bantuan konselor profesional. "
            "Ini adalah respons penutup terakhir Anda. Berikan penutupan yang hangat, penuh kasih, dan bermakna. "
            "Ringkas perjalanan konseling hari ini dengan singkat. "
            "Ucapkan selamat jalan, sampaikan doa dan harapan terbaik untuk pengguna. "
            "Tegaskan bahwa mereka bisa kembali kapan saja jika ingin bercerita lagi. "
            "JANGAN menawarkan bantuan konselor profesional lagi. "
            "JANGAN bertanya apapun lagi — ini adalah akhir sesi."
        ),
    }

    # Additional CBT-based guidance injected into solusi/relaksasi prompts
    # when the patient has shown physical symptoms (Mengisyaratkan Gejala Fisik)
    PHYSICAL_SYMPTOM_CBT_GUIDANCE = (
        "Selain itu, pengguna juga menunjukkan gejala fisik (seperti gangguan tidur, "
        "kelelahan berlebihan, sesak napas, atau perubahan nafsu makan) yang sering "
        "dipicu oleh kecemasan, pikiran berlebihan, dan perubahan hormonal. "
        "Sarankan teknik-teknik coping praktis dari Cognitive Behavioral Therapy (CBT): "
        "latihan pernapasan dalam (deep breathing), teknik mindfulness, dan strategi "
        "relaksasi tubuh. Dorong juga aktivasi perilaku \u2014 ajak pengguna melakukan "
        "aktivitas positif dan bertujuan untuk memulihkan energi fisik dan memperbaiki "
        "suasana hati secara keseluruhan."
    )

    # Spiritual-consent question injected at the FINAL turn of the solusi stage.
    # Verse injection is only active in relaksasi, so asking at the last solusi turn
    # ensures consent is resolved before relaksasi begins. The strict constraint
    # ("HANYA boleh menanyakan") prevents the LLM from mixing the ask with other content,
    # reducing misinterpretation of the user's yes/no reply.
    SPIRITUAL_CONSENT_PROMPT = (
        "Pada giliran ini, Anda HANYA boleh menanyakan kesediaan klien untuk menggunakan Alkitab. "
        "DILARANG mengajukan pertanyaan lain agar klien fokus menjawab ya atau tidak. "
        "Tanyakan dengan lembut dan persis seperti ini: "
        "'Apakah Anda bersedia melihat masalah ini dari sudut pandang firman Tuhan atau Alkitab?'"
    )

    # Stages where bible verses should be retrieved and injected.
    # Restricted to relaksasi only — solusi and bantuan_profesional no longer
    # receive verse injection (solusi now only asks for consent; bantuan_profesional
    # removed to avoid spiritual imposition in a crisis branch).
    BIBLE_VERSE_STAGES = ['relaksasi']

    # Prompt injected at pembahasan turn >= 3 to ask if the user has anything
    # else to share. Strictly forbids solutions, conclusions, or spiritual topics.
    # Applied on top of the base pembahasan instruction.
    PEMBAHASAN_TURN3_PROMPT = (
        "Anda telah mengeksplorasi masalah klien. Pada giliran ini, Anda HARUS bertanya kembali "
        "kepada klien apakah masih ada hal lain yang ingin diceritakan. "
        "DILARANG KERAS memberikan solusi, kesimpulan, atau menyinggung tentang Alkitab dan "
        "spiritualitas pada tahap ini."
    )

    # Stages where intent classification can be safely skipped (no verse retrieval,
    # accumulated intents from these stages don't feed primary_intents)
    SKIP_CLASSIFICATION_STAGES = frozenset({'pembukaan', 'bantuan_profesional'})

    def __init__(self):
        # Initialize the LLM (provider set in config.py → opt.CHATBOT_LLM_PROVIDER)
        self.llm = _build_chatbot_llm()

        # Initialize the Multilabel Classifier (Predictor)
        self.classifier = predictor()

        # Initialize the Vector DB Manager (Knowledge Base)
        self.vector_db = VectorDBManager()
        # Build indices if not built
        self.vector_db.build_bible_index('data/verse_retrieval/alkitab_tb_enriched_groq.csv')
        self.vector_db.build_qna_index('data/dataset_qna.csv')

        # Define the Prompt Template for the LLM
        # Note: stage_instruction replaces the explicit stage label for natural conversation
        # bible_section is conditionally included only during relaksasi/solusi stages
        self.prompt_template = PromptTemplate(
            input_variables=[
                "user_input",
                "detected_intents",
                "stage_instruction",
                "example_answer",
                "bible_section"
            ],
            template="""Anda adalah seorang psikolog dan konselor Kristen yang berempati, hangat, dan bijaksana.
Tugas Anda adalah merespons curhatan atau pertanyaan pengguna dengan cara yang suportif dan natural.

Informasi Konteks:
- Panduan Perilaku Anda Saat Ini: {stage_instruction}
- Input Pengguna: "{user_input}"
- Emosi/Niat Terdeteksi (Multilabel): {detected_intents}

Bahan Inspirasi dari Knowledge Base:
- Contoh Respons Terdahulu (Gunakan sebagai inspirasi nada dan konten): 
"{example_answer}"

{bible_section}

Instruksi:
1. Berikan respons yang terasa natural, berempati, dan seperti percakapan sungguhan dengan psikolog.
2. Ikuti "Panduan Perilaku" yang diberikan untuk menentukan pendekatan dan nada bicara Anda.
3. Gunakan "Contoh Respons Terdahulu" sebagai inspirasi untuk cara menjawab yang baik, tetapi susun ulang dengan bahasa Anda sendiri yang lebih natural.
4. Jika ada bagian "Ayat Alkitab Relevan", jadikan ayat tersebut sebagai sumber kekuatan dan penghiburan, jelaskan dengan lembut bagaimana ayat itu bisa menguatkan pengguna. Jangan menghakimi.
5. Jika TIDAK ada bagian "Ayat Alkitab Relevan", jangan menambahkan ayat Alkitab sendiri.
6. Jangan menyebutkan bahwa Anda adalah AI atau bot. Bertindaklah seperti konselor manusia yang peduli.
7. Jangan menyebutkan nama tahapan konseling apapun (seperti "pembahasan", "intervensi", dll).

Respons Anda:
"""
        )

        # Create the LangChain processing chain
        self.chain = self.prompt_template | self.llm

    def generate_background_summary(self, history_list):
        """
        Summarize the pembahasan conversation history into one sentence.
        Called once at the pembahasan→intervensi transition by SessionManager.
        The result is stored on the session and injected into later-stage prompts.

        Args:
            history_list: list of (user_message, bot_response) tuples from pembahasan.

        Returns:
            str: One-sentence summary of the client's core complaint, or "" on failure.
        """
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
        """
        Extract the relaxation technique the client chose during the solusi stage.
        Called once at the solusi→relaksasi transition by SessionManager.
        The result is stored on the session and injected into the relaksasi prompt
        so the LLM guides the user through their chosen technique without re-listing all options.

        Args:
            history_list: list of (user_message, bot_response) tuples from solusi.

        Returns:
            str: Short technique name (e.g. "Pernapasan 4-7-8"), or None if not found.
        """
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
        """
        Generate a counseling response.

        Args:
            user_input: The user's message text.
            current_stage: The current counseling stage.
            override_intents: Optional list of intents to use for bible verse retrieval
                              instead of the current turn's detected intents. This allows
                              the SessionManager to pass accumulated intents from earlier
                              stages (e.g., pembahasan) for more relevant verse selection.
            has_physical_symptoms: If True, inject CBT-based physical symptom guidance
                                   into solusi/relaksasi stage instructions.
            excluded_books: Optional set of book_abbr strings to exclude from verse
                            retrieval (session-level book exclusion).
            excluded_verses: Optional set of exact reference strings to exclude
                             (e.g. {"Mazmur 34:18"}) — session-level verse exclusion.
            spiritual_consent: Tri-state spiritual consent (True/False/None). Bible verses
                               are only retrieved and injected when this is explicitly True.
                               None (not yet answered) or False both bypass verse retrieval,
                               yielding pure psychological counseling (no value imposition).
            ask_spiritual_consent: If True, append the spiritual-consent question to the
                                   intervensi prompt so the LLM asks the user's permission
                                   before any Scripture is introduced.
            turn_in_stage: How many turns have elapsed in the current stage (1-indexed).
                           Used to conditionally inject the pembahasan turn-3+ confirmation
                           prompt that asks if the user has anything else to share.
            complaint_summary: One-sentence summary of the client's core complaint,
                               generated at the pembahasan→intervensi transition. Injected
                               as context prefix in intervensi, solusi, and relaksasi prompts
                               so the LLM retains awareness of the original issue despite
                               being stateless across turns.
            chosen_technique: Relaxation technique the client selected during solusi
                              (e.g. "Pernapasan 4-7-8"), extracted at the solusi→relaksasi
                              transition. Injected as a prefix into the relaksasi instruction
                              so the LLM guides only the chosen technique, not all options.
        """
        # Get stage-specific behavioral instruction
        stage_instruction = self.STAGE_INSTRUCTIONS.get(
            current_stage,
            self.STAGE_INSTRUCTIONS["pembahasan"]  # fallback
        )

        # Inject ephemeral complaint summary into later stages so the LLM retains
        # awareness of the client's core issue across the stateless turn boundary.
        if complaint_summary and current_stage in ('intervensi', 'solusi', 'relaksasi'):
            stage_instruction = f"Konteks Keluhan Klien: {complaint_summary}\n\n" + stage_instruction

        # Inject the chosen relaxation technique for the relaksasi stage so the LLM
        # only guides the agreed technique instead of re-listing all options.
        if chosen_technique and current_stage == 'relaksasi':
            stage_instruction = (
                f"Klien telah memilih teknik {chosen_technique}. "
                f"Pandu klien HANYA dengan teknik tersebut. "
            ) + stage_instruction

        # Pembahasan turn-3+ injection: once 3 exploration turns have elapsed,
        # the LLM must explicitly ask if the user has anything else to share.
        # Turns 1 and 2 retain the base constraint (no solutions, open questions only).
        if current_stage == 'pembahasan' and turn_in_stage >= 3:
            stage_instruction = stage_instruction + " " + self.PEMBAHASAN_TURN3_PROMPT

        # Inject CBT physical symptom guidance when applicable
        if has_physical_symptoms and current_stage in ('solusi', 'relaksasi'):
            stage_instruction = stage_instruction + " " + self.PHYSICAL_SYMPTOM_CBT_GUIDANCE

        # Ask for spiritual consent at the FINAL turn of the solusi stage so the
        # user can give a clean yes/no without it being mixed with exploration questions.
        # Verse injection only starts in relaksasi, so this timing ensures consent
        # is resolved before any Scripture is ever introduced.
        if ask_spiritual_consent and current_stage == 'solusi':
            stage_instruction = stage_instruction + " " + self.SPIRITUAL_CONSENT_PROMPT

        _t0 = time.perf_counter()

        # 1. Intent classification — skipped in stages where it has no effect on
        # the response (no bible retrieval, intents not fed into primary_intents)
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

        # 2. Get Example Answer from QnA DB
        qna_results = self.vector_db.search_qna(user_input, k=1)
        example_answer = ""
        if qna_results:
            example_answer = qna_results[0].metadata.get('answer', '')
        _t2 = time.perf_counter()
        print(f"[TIMING] search_qna: {(_t2 - _t1) * 1000:.1f}ms")

        # 3. Get Bible Verses — ONLY during relaksasi (primary) and solusi (conditional)
        # This prevents premature spiritual guidance before the user has shared their burden.
        # When override_intents are provided (from SessionManager's accumulated intents),
        # use them instead of the current turn's intents for more relevant verse selection.
        # ALL intents are combined together to determine the 1 best verse.
        # Uses LLM reranker: FAISS retrieves top-10 candidates, Gemini picks the best one.
        bible_verses = ""
        bible_reference = ""
        bible_book_abbr = ""
        # Conditional RAG: verses are only retrieved when the user has explicitly
        # given spiritual consent (True). Without it, we bypass RAG entirely and
        # provide pure psychological counseling — no spiritual value imposition.
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

        # 4. Build the bible section dynamically
        # Only include in the prompt if we actually have a verse
        bible_section = ""
        if bible_verses:
            verse_display = f"{bible_reference} (TB) \"{bible_verses}\""
            bible_section = (
                "- Ayat Alkitab Relevan (Sertakan dan jelaskan maknanya dengan lembut dalam konteks masalah pengguna):\n"
                f'{verse_display}'
            )

        # 5. Format the inputs and invoke the LLM Chain
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

        # Return the generated text and the context used (for development/debugging)
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
