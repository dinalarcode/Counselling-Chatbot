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
            "Dengarkan dengan empati dan validasi perasaan mereka. "
            "Jika masih kurang jelas, ajukan pertanyaan terbuka untuk memahami situasi. "
            "Namun, jika pengguna sudah menceritakan inti masalahnya atau terlihat sudah selesai, "
            "jangan terus memaksa mereka bercerita. Tanyakan secara halus apakah mereka siap "
            "untuk mendiskusikan pandangan lain atau melangkah maju. "
            "Fokus pada mendengarkan, belum memberikan solusi atau ayat Alkitab."
            "Jangan bertele-tele dalam menanggapi."
        ),
        "intervensi": (
            "Anda sudah memahami masalah pengguna dan sekarang saatnya memberikan perspektif baru. "
            "Berikan pandangan yang berempati dan bantu pengguna melihat situasi dari sudut pandang berbeda. "
            "Reframe pemikiran negatif menjadi lebih konstruktif. "
            "Belum perlu menyertakan ayat Alkitab di sini, fokus pada pendekatan psikologis yang hangat."
        ),
        "solusi": (
            "Bantu pengguna merumuskan langkah-langkah konkret yang bisa mereka ambil. "
            "Berikan saran praktis yang realistis dan dorong mereka untuk bertindak. "
            "Jika ada ayat Alkitab yang relevan dan mendukung langkah tersebut, "
            "sebutkan secara lembut sebagai penguat \u2014 bukan menghakimi, "
            "melainkan sebagai sumber kekuatan dan inspirasi."
            "Berikan jawaban yang singkat, padat, dan bermakna"
        ),
        "relaksasi": (
            "Bantu pengguna untuk menenangkan diri setelah diskusi yang panjang. "
            "Gunakan ayat Alkitab yang relevan untuk menenangkan dan menguatkan hati pengguna. "
            "Jelaskan makna ayat tersebut dengan lembut dalam konteks perasaan dan masalah pengguna. "
            "Ajak mereka untuk merenungkan firman Tuhan dan merasakan ketenangan dari-Nya. "
            "Gunakan nada yang menenangkan, penuh kasih, dan perlahan."
            "Tanyakan kepada pengguna apakah mereka sudah merasa terberkati atau menunjukan tanda paham dengan firman Tuhan."
            "Jika sudah jangan diberikan ayat terus menerus."
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
            "Gunakan ayat Alkitab yang diberikan untuk menguatkan bahwa Tuhan mendukung mereka dalam langkah ini. "
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

    # Stages where bible verses should be retrieved and injected
    BIBLE_VERSE_STAGES = ['relaksasi', 'solusi', 'bantuan_profesional']

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

    def generate_response(self, user_input, current_stage="pembahasan", override_intents=None, has_physical_symptoms=False, excluded_books=None, excluded_verses=None):
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
        """
        # Get stage-specific behavioral instruction
        stage_instruction = self.STAGE_INSTRUCTIONS.get(
            current_stage,
            self.STAGE_INSTRUCTIONS["pembahasan"]  # fallback
        )

        # Inject CBT physical symptom guidance when applicable
        if has_physical_symptoms and current_stage in ('solusi', 'relaksasi'):
            stage_instruction = stage_instruction + " " + self.PHYSICAL_SYMPTOM_CBT_GUIDANCE

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
        intents_for_verse = override_intents if override_intents else detected_intents
        if current_stage in self.BIBLE_VERSE_STAGES and intents_for_verse:
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
