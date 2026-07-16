"""Static data for the vector DB manager including stopwords, biblical synonyms, and diversity constants."""

# Maps each of the 10 LABAN intents to formal TB Bible vocabulary injected into the FAISS query for expansion.
BIBLICAL_SYNONYMS = {
    "Mengisyaratkan Butuh Bantuan Profesional": (
        "pertolongan penolong selamatkan tolong lindungi "
        "perlindungan harapan kekuatan penghiburan pemulihan"
        "nasihat didikan hikmat teguran petunjuk jalan keluar terang kekuatan"
    ),
    "Mengisyaratkan Gejala Fisik": (
        "sakit penyakit lemah lelah tubuh daging "
        "luka menderita penderitaan kesembuhan"
        "lesu penat letih sembuh menyegarkan membalut kekuatan baru memulihkan"
    ),
    "Menyatakan Perasaan Benci dan Jijik": (
        "benci kebencian murka jijik kejijikan hina "
        "menghina najis kekejian muak"
        "kekejian muak mengampuni kasih damai memberkati pengampunan saudara"
    ),
    "Menyatakan Perasaan Marah dan Frustasi": (
        "murka amarah marah gusar geram berang "
        "kemarahan emosi mengeluh keluh kesah"
        "dendam sabar kasih karunia menahan diri damai sejahtera lemah lembut"
    ),
    "Menyatakan Perasaan Percaya": (
        "percaya iman beriman setia kesetiaan "
        "pengharapan berharap yakin teguh penyertaan"
        "berserah percaya setia dijagai janji aman tempat perlindungan"
    ),
    "Menyatakan Perasaan Sebelum Menghadapi Kejadian": (
        "kuatir khawatir gelisah waswas gentar "
        "bimbang ragu cemas menanti menunggu"
        "berjaga-jaga pencobaan ujian waspada penyertaan jangan takut berani teguh melangkah"

    ),
    "Menyatakan Perasaan Sedih dan Kehilangan": (
        "dukacita ratap tangis meratap bersedih perkabungan "
        "berkabung kehilangan air mata patah hati remuk jiwa"
        "duka ratapan hancur hati penghiburan dihibur sukacita memulihkan"
    ),
    "Menyatakan Perasaan Takut dan Kecemasan": (
        "takut ketakutan gentar ngeri kecemasan gemetar "
        "cemas waswas kuatir gelisah"
        "damai berani aman perlindungan tempat perlindungan penolong"
    ),
    "Menyatakan Rasa Syukur dan Apresiasi": (
        "syukur bersyukur puji pujian memuji ucapan syukur "
        "berkat diberkati terima kasih sukacita"
        "sukacita sorak karunia kebaikan melimpah"
    ),
    "Menyatakan Reaksi Terkejut dan Tidak Terduga": (
        "heran takjub tercengang dahsyat ajaib mujizat "
        "keajaiban terkejut terperanjat"
        "gempar kedaulatan rencana damai sejahtera pegangan pertolongan ajaib"
    ),
}

# Diversity constants used by _faiss_search to ensure candidates come from varied chapters.
OVERSAMPLE_FACTOR = 3     # Fetch three times more from FAISS before filtering.
MAX_PER_CHAPTER   = 2     # Keep at most two verses per chapter in the final k candidates.
