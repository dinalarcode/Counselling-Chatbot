import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load the predictor and VectorDBManager
from models.multilabel.predict import predictor
from core.vector_db import VectorDBManager

# Load environment variables
load_dotenv()

class RAGEngine:
    def __init__(self, model_name="gemini-1.5-flash", temperature=0.7):
        # Retrieve the API key from environment variables
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Please set it in the .env file.")

        # Initialize the LLM (Gemini)
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=temperature
        )

        # Initialize the Multilabel Classifier (Predictor)
        self.classifier = predictor()

        # Initialize the Vector DB Manager (Knowledge Base)
        self.vector_db = VectorDBManager()
        # Build indices if not built
        self.vector_db.build_ayat_index('data/dataset_ayat.csv')
        self.vector_db.build_qna_index('data/dataset_qna.csv')

        # Define the Prompt Template for the LLM
        self.prompt_template = PromptTemplate(
            input_variables=[
                "user_input",
                "detected_intents",
                "current_stage",
                "example_answer",
                "bible_verses"
            ],
            template="""Anda adalah seorang psikolog dan konselor Kristen yang berempati, hangat, dan bijaksana.
Tugas Anda adalah merespons curhatan atau pertanyaan pengguna dengan cara yang suportif, mengikuti tahapan konseling saat ini.

Informasi Konteks:
- Tahapan Konseling Saat Ini: {current_stage}
- Input Pengguna: "{user_input}"
- Emosi/Niat Terdeteksi (Multilabel): {detected_intents}

Bahan Inspirasi dari Knowledge Base:
- Contoh Respons Terdahulu (Gunakan sebagai inspirasi nada dan konten): 
"{example_answer}"

- Ayat Alkitab Relevan (Sertakan dan jelaskan maknanya jika sesuai dengan konteks): 
"{bible_verses}"

Instruksi:
1. Berikan respons yang terasa natural, berempati, dan seperti percakapan sungguhan dengan psikolog.
2. Sesuaikan respons Anda dengan "Tahapan Konseling Saat Ini".
3. Gunakan "Contoh Respons Terdahulu" sebagai inspirasi untuk cara menjawab yang baik, tetapi susun ulang dengan bahasa Anda sendiri yang lebih natural.
4. Jika ada "Ayat Alkitab Relevan", jadikan ayat tersebut sebagai sumber kekuatan atau intervensi, jelaskan dengan lembut bagaimana ayat itu bisa menguatkan pengguna.
5. Jangan menyebutkan bahwa Anda adalah AI atau bot. Bertindaklah seperti konselor manusia yang peduli.

Respons Anda:
"""
        )

        # Create the LangChain processing chain
        self.chain = self.prompt_template | self.llm

    def generate_response(self, user_input, current_stage="pembahasan"):
        # 1. Get Intent Predictions
        prediction = self.classifier.predict(user_input)
        detected_intents = prediction.get('inten_terdeteksi', [])
        intents_str = ", ".join(detected_intents) if detected_intents else "Umum"

        # 2. Get Example Answer from QnA DB
        qna_results = self.vector_db.search_qna(user_input, k=1)
        example_answer = ""
        if qna_results:
            example_answer = qna_results[0].metadata.get('answer', '')

        # 3. Get Bible Verses based on Detected Intents
        bible_verses = ""
        if detected_intents:
            primary_intent = detected_intents[0]
            ayat_results = self.vector_db.get_ayat_by_intent(primary_intent, k=1)
            if ayat_results:
                bible_verses = ayat_results[0].metadata.get('ayat', '')

        # 4. Format the inputs and invoke the LLM Chain
        response = self.chain.invoke({
            "user_input": user_input,
            "detected_intents": intents_str,
            "current_stage": current_stage,
            "example_answer": example_answer,
            "bible_verses": bible_verses
        })

        # Return the generated text and the context used
        return {
            "response": response.content,
            "context_used": {
                "intents": detected_intents,
                "example_answer": example_answer,
                "bible_verses": bible_verses,
                "stage": current_stage
            }
        }
