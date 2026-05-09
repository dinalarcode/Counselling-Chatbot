import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load the predictor and VectorDBManager
from models.multilabel.predict import predictor
from core.vector_db import VectorDBManager

# Load environment variables
load_dotenv()

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
        ),
        "pembahasan": (
            "Pengguna sedang menceritakan masalahnya. "
            "Dengarkan dengan empati dan validasi perasaan mereka. "
            "Jika masih kurang jelas, ajukan pertanyaan terbuka untuk memahami situasi. "
            "Namun, jika pengguna sudah menceritakan inti masalahnya atau terlihat sudah selesai, "
            "jangan terus memaksa mereka bercerita. Tanyakan secara halus apakah mereka siap "
            "untuk mendiskusikan pandangan lain atau melangkah maju. "
            "Fokus pada mendengarkan, belum memberikan solusi atau ayat Alkitab."
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
            "sebutkan secara lembut sebagai penguat — bukan menghakimi, "
            "melainkan sebagai sumber kekuatan dan inspirasi."
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
            "Jangan menyertakan ayat Alkitab baru, cukup berikan penutupan yang hangat."
        ),
    }

    # Stages where bible verses should be retrieved and injected
    BIBLE_VERSE_STAGES = ['relaksasi', 'solusi']

    def __init__(self, model_name="deepseek-v4-pro", temperature=0.7):
        # Retrieve the API key from environment variables
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Please set it in the .env file.")

        # Initialize the LLM (Gemini)
        # self.llm = ChatGoogleGenerativeAI(
        #     model=model_name,
        #     google_api_key=self.api_key,
        #     temperature=temperature
        # )
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com",
            temperature=temperature
        )

        # Initialize the Multilabel Classifier (Predictor)
        self.classifier = predictor()

        # Initialize the Vector DB Manager (Knowledge Base)
        self.vector_db = VectorDBManager()
        # Build indices if not built
        self.vector_db.build_bible_index('data/alkitab_tb.csv')
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

    def generate_response(self, user_input, current_stage="pembahasan", override_intents=None):
        """
        Generate a counseling response.

        Args:
            user_input: The user's message text.
            current_stage: The current counseling stage.
            override_intents: Optional list of intents to use for bible verse retrieval
                              instead of the current turn's detected intents. This allows
                              the SessionManager to pass accumulated intents from earlier
                              stages (e.g., pembahasan) for more relevant verse selection.
        """
        # Get stage-specific behavioral instruction
        stage_instruction = self.STAGE_INSTRUCTIONS.get(
            current_stage,
            self.STAGE_INSTRUCTIONS["pembahasan"]  # fallback
        )

        # 1. Get Intent Predictions (always run for the current turn)
        prediction = self.classifier.predict(user_input)
        detected_intents = prediction.get('inten_terdeteksi', [])
        intents_str = ", ".join(detected_intents) if detected_intents else "Umum"

        # 2. Get Example Answer from QnA DB
        qna_results = self.vector_db.search_qna(user_input, k=1)
        example_answer = ""
        if qna_results:
            example_answer = qna_results[0].metadata.get('answer', '')

        # 3. Get Bible Verses — ONLY during relaksasi (primary) and solusi (conditional)
        # This prevents premature spiritual guidance before the user has shared their burden.
        # When override_intents are provided (from SessionManager's accumulated intents),
        # use them instead of the current turn's intents for more relevant verse selection.
        # ALL intents are combined together to determine the 1 best verse.
        bible_verses = ""
        bible_reference = ""
        intents_for_verse = override_intents if override_intents else detected_intents
        if current_stage in self.BIBLE_VERSE_STAGES and intents_for_verse:
            verse_results = self.vector_db.retrieve_verse(
                intents=intents_for_verse,
                user_input=user_input,
                k=1
            )
            if verse_results:
                bible_reference = verse_results[0].get('reference', '')
                bible_verses = verse_results[0].get('text', '')

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

        # Return the generated text and the context used (for development/debugging)
        return {
            "response": response.content,
            "context_used": {
                "intents": detected_intents,
                "example_answer": example_answer,
                "bible_verses": f"{bible_reference} - {bible_verses}" if bible_verses else "",
                "stage": current_stage
            }
        }
