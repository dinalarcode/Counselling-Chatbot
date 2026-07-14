"""Static data lists and dicts for the counseling SessionManager stage machine."""

import re

# Ordered counseling stages in the normal linear flow.
STAGE_ORDER = ['pembukaan', 'pembahasan', 'intervensi', 'solusi', 'relaksasi', 'penutupan']

# Branch stage accessed via emergency skip or the penutupan referral offer.
PROFESSIONAL_STAGE = 'bantuan_profesional'

# Intent name that triggers the professional help branch.
PROFESSIONAL_INTENT = 'Mengisyaratkan Butuh Bantuan Profesional'
# Intent name that flags physical symptoms across the session.
PHYSICAL_SYMPTOM_INTENT = 'Mengisyaratkan Gejala Fisik'

# Keyword safety net catching suicidal or hopeless expressions the classifier may miss.
PROFESSIONAL_KEYWORDS = [
    # Suicidal ideation patterns.
    re.compile(r'\bmati\s*(aja|saja)\b', re.IGNORECASE),
    re.compile(r'\bbunuh\s*diri\b', re.IGNORECASE),
    re.compile(r'\bmengakhiri\s*(hidup|hidupku|nyawa)\b', re.IGNORECASE),
    re.compile(r'\bgantung\s*diri\b', re.IGNORECASE),
    re.compile(r'\bloncat\b.*\b(gedung|jembatan)\b', re.IGNORECASE),
    re.compile(r'\bmending\s*mati\b', re.IGNORECASE),
    re.compile(r'\b(lebih\s*baik|mending)\s*(ga|nggak|tidak)\s*ada\b', re.IGNORECASE),
    re.compile(r'\b(gamau|ga\s*mau|nggak\s*mau|tidak\s*mau)\s*hidup\b', re.IGNORECASE),
    # Hopelessness or giving up patterns.
    re.compile(r'\bga\s*ada\s*(gunanya|artinya|tujuan)\b', re.IGNORECASE),
    re.compile(r'\b(gaada|ga\s*ada)\s*(harapan|masa\s*depan)\b', re.IGNORECASE),
    re.compile(r'\blebih\s*baik\s*(aku|saya|gue)\s*(pergi|hilang|mati)\b', re.IGNORECASE),
    re.compile(r'\bselesai(kan)?\s*(hidup|semua)\b', re.IGNORECASE),
    re.compile(r'\b(udah|sudah|aku|saya|gue|gw)\s+(nyerah|menyerah)\b', re.IGNORECASE),
    re.compile(r'\bgak?\s*berarti\b', re.IGNORECASE),
    # Self-harm patterns.
    re.compile(r'\b(nyakitin|menyakiti|melukai)\s*diri\b', re.IGNORECASE),
    re.compile(r'\b(iris|sayat|potong)\s*(tangan|nadi|pergelangan)\b', re.IGNORECASE),
]

# Decline signals checked only at penutupan when the user refuses the referral offer.
DECLINE_SIGNALS = [
    re.compile(r'\btidak\b', re.IGNORECASE),
    re.compile(r'\bnggak\b', re.IGNORECASE),
    re.compile(r'\bgak\b', re.IGNORECASE),
    re.compile(r'\bga\b', re.IGNORECASE),
    re.compile(r'\bsudah\s+cukup\b', re.IGNORECASE),
    re.compile(r'\bcukup\b', re.IGNORECASE),
    re.compile(r'\btidak\s+perlu\b', re.IGNORECASE),
    re.compile(r'\btidak\s+usah\b', re.IGNORECASE),
    re.compile(r'\bnggak\s+perlu\b', re.IGNORECASE),
    re.compile(r'\bterima\s*kasih\b', re.IGNORECASE),
    re.compile(r'\bmakasih\b', re.IGNORECASE),
]

# Decline patterns for the spiritual consent question, checked before the affirmative set.
SPIRITUAL_CONSENT_DECLINE = [
    re.compile(r'\btidak\b', re.IGNORECASE),
    re.compile(r'\bnggak\b', re.IGNORECASE),
    re.compile(r'\benggak\b', re.IGNORECASE),
    re.compile(r'\bjangan\b', re.IGNORECASE),
    re.compile(r'\bbelum\b', re.IGNORECASE),
    re.compile(r'\bnanti\s+(saja|aja|dulu)\b', re.IGNORECASE),
    re.compile(r'\bga\s+(mau|usah|perlu)\b', re.IGNORECASE),
    re.compile(r'\bgak\s+(mau|usah|perlu)\b', re.IGNORECASE),
]
# Affirmative patterns for the spiritual consent question.
SPIRITUAL_CONSENT_AFFIRM = [
    re.compile(r'\b(iya|ya|yah|yaudah|yauda)\b', re.IGNORECASE),
    re.compile(r'\bmau\b', re.IGNORECASE),
    re.compile(r'\bboleh\b', re.IGNORECASE),
    re.compile(r'\bbersedia\b', re.IGNORECASE),
    re.compile(r'\btentu\b', re.IGNORECASE),
    re.compile(r'\bsila(h)?kan\b', re.IGNORECASE),
    re.compile(r'\bsetuju\b', re.IGNORECASE),
    re.compile(r'\b(oke|ok|okay|oce)\b', re.IGNORECASE),
    re.compile(r'\b(ayo|mari)\b', re.IGNORECASE),
    re.compile(r'\bbaik(lah)?\b', re.IGNORECASE),
    re.compile(r'\b(firman|alkitab|ayat|tuhan|rohani)\b', re.IGNORECASE),
]

# Minimum turns required before a stage can transition.
MIN_TURNS = {
    'pembukaan': 1,
    'pembahasan': 3,  # Clinical requirement of at least 3 exploration turns.
    'intervensi': 1,
    'solusi': 1,
    'relaksasi': 1,
    'penutupan': 1,
}

# Maximum turns that force a transition as a safety net.
MAX_TURNS = {
    'pembukaan': 2,
    'pembahasan': 5,  # Extended ceiling to allow richer exploration.
    'intervensi': 3,
    'solusi': 3,
    'relaksasi': 3,
    'penutupan': 99,  # Never force-exit penutupan.
}

# Phrases signaling the user has nothing more to share, bypassing the pembahasan minimum.
PEMBAHASAN_EARLY_EXIT_SIGNALS = [
    re.compile(r'\btidak\s+ada\b', re.IGNORECASE),
    re.compile(r'\benggak\s+ada\b', re.IGNORECASE),
    re.compile(r'\bnggak\s+ada\b', re.IGNORECASE),
    re.compile(r'\bgak\s+ada\b', re.IGNORECASE),
    re.compile(r'\bga\s+ada\b', re.IGNORECASE),
    re.compile(r'\bsudah\s+cukup\b', re.IGNORECASE),
    re.compile(r'\bsudah\s+cukup\s+cerita\b', re.IGNORECASE),
    re.compile(r'\bcuma\s+itu\b', re.IGNORECASE),
    re.compile(r'\bcuma\s+itu\s+saja\b', re.IGNORECASE),
    re.compile(r'\bhanya\s+itu\b', re.IGNORECASE),
    re.compile(r'\bitu\s+saja\b', re.IGNORECASE),
    re.compile(r'\bitu\s+aja\b', re.IGNORECASE),
    re.compile(r'\btidak\s+ada\s+lagi\b', re.IGNORECASE),
    re.compile(r'\bnggak\s+ada\s+lagi\b', re.IGNORECASE),
    re.compile(r'\bgak\s+ada\s+lagi\b', re.IGNORECASE),
    re.compile(r'\bga\s+ada\s+lagi\b', re.IGNORECASE),
    re.compile(r'\bsegitu\s+saja\b', re.IGNORECASE),
    re.compile(r'\bsegitu\s+aja\b', re.IGNORECASE),
    re.compile(r'\bselesai\b', re.IGNORECASE),
    re.compile(r'\bsudah\s+selesai\b', re.IGNORECASE),
]

# Transition signal patterns per stage in Indonesian phrases, matched after the minimum turns.
TRANSITION_SIGNALS = {
    'pembukaan': [
        # Pembukaan auto-advances via MIN_TURNS of 1 so no signals are needed.
    ],
    'pembahasan': [
        r'\bsudah\s+cukup\b',
        r'\btidak\s+ada\s+lagi\b',
        r'\bnggak\s+ada\s+lagi\b',
        r'\bgak\s+ada\s+lagi\b',
        r'\bitu\s+saja\b',
        r'\bitu\s+aja\b',
        r'\bcukup\b',
        r'\blanjut\b',
        r'\bmari\s+lanjut\b',
        r'\bkita\s+lanjut\b',
        r'\btidak\b.*\blagi\b',
        r'\bnggak\b.*\blagi\b',
        r'^tidak$',
        r'^nggak$',
        r'^gak$',
        r'^udah$',
        r'^sudah$',
        r'\bsudah\b',
        r'\bhanya\s+itu\b',
        r'\bsepertinya\s+sudah\b',
        r'\bbisa\b',
        r'\bboleh\b',
        r'\bsiap\b',
    ],
    'intervensi': [
        r'\bsaya\s+mengerti\b',
        r'\baku\s+mengerti\b',
        r'\bbenar\s+juga\b',
        r'\bmasuk\s+akal\b',
        r'\bsetuju\b',
        r'\biya\b',
        r'\bbaik\b',
        r'\bbenar\b',
        r'\boh\s+iya\b',
        r'\bmemang\s+benar\b',
        r'\bsaya\s+paham\b',
        r'\baku\s+paham\b',
    ],
    'solusi': [
        r'\bakan\s+(saya|aku)\s+coba\b',
        r'\bsaya\s+coba\b',
        r'\baku\s+coba\b',
        r'\bterima\s*kasih\b',
        r'\bmakasih\b',
        r'\boke\b',
        r'\bbaik\b',
        r'\bsaya\s+akan\b',
        r'\baku\s+akan\b',
        r'\bbisa\s+dicoba\b',
        r'\bmari\b',
        # The broad pattern bisa was removed to avoid false transitions on phrases like bisa jadi.
    ],
    'relaksasi': [
        r'\blebih\s+tenang\b',
        r'\blebih\s+baik\b',
        r'\bterima\s*kasih\b',
        r'\bmakasih\b',
        r'\biya\b',
        r'\bsudah\b',
        r'\bbaik\b',
        r'\bsedikit\s+lega\b',
        r'\blega\b',
        r'\bamin\b',
        r'\bberkat\b',
    ],
    'penutupan': [
        # Signals for accepting the professional referral offer.
        r'\biya\b',
        r'\bmau\b',
        r'\bboleh\b',
        r'\boke\b',
        r'\bbaik\b',
        r'\bsiap\b',
        r'\blanjut\b',
        r'^ya$',
    ],
}

# Pre-compiled transition signal patterns built once from TRANSITION_SIGNALS for fast matching.
_COMPILED_SIGNALS = {
    stage: [re.compile(p) for p in patterns]
    for stage, patterns in TRANSITION_SIGNALS.items()
}
