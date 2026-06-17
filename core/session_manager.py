"""
Session Manager — Automatic Stage Progression for Counseling Chatbot.

This module manages the counseling session state, including:
- Stage tracking and automatic progression
- Turn counting with minimum thresholds per stage
- Transition signal detection from user messages
- Intent accumulation across the conversation for context-aware bible verse retrieval
"""

import re
from collections import Counter


class SessionManager:
    """
    Wraps the RAG engine and manages counseling session state.
    Automatically transitions between stages based on turn count
    and transition signals detected in user messages.
    """

    # Ordered counseling stages (normal linear flow)
    STAGE_ORDER = ['pembukaan', 'pembahasan', 'intervensi', 'solusi', 'relaksasi', 'penutupan']

    # Branch stage — not part of the linear flow; accessed via emergency skip
    # or penutupan referral offer
    PROFESSIONAL_STAGE = 'bantuan_profesional'

    # Intent names that trigger special behavior
    PROFESSIONAL_INTENT = 'Mengisyaratkan Butuh Bantuan Profesional'
    PHYSICAL_SYMPTOM_INTENT = 'Mengisyaratkan Gejala Fisik'

    # Keyword-based safety net for professional help detection.
    # These catch suicidal/hopeless expressions that the classifier may miss.
    # Compiled once at class load for fast matching.
    PROFESSIONAL_KEYWORDS = [
        # Suicidal ideation
        re.compile(r'\bmati\s*(aja|saja)\b', re.IGNORECASE),
        re.compile(r'\bbunuh\s*diri\b', re.IGNORECASE),
        re.compile(r'\bmengakhiri\s*(hidup|hidupku|nyawa)\b', re.IGNORECASE),
        re.compile(r'\bgantung\s*diri\b', re.IGNORECASE),
        re.compile(r'\bloncat\b.*\b(gedung|jembatan)\b', re.IGNORECASE),
        re.compile(r'\bmending\s*mati\b', re.IGNORECASE),
        re.compile(r'\b(lebih\s*baik|mending)\s*(ga|nggak|tidak)\s*ada\b', re.IGNORECASE),
        re.compile(r'\b(gamau|ga\s*mau|nggak\s*mau|tidak\s*mau)\s*hidup\b', re.IGNORECASE),
        # Hopelessness / giving up
        re.compile(r'\bga\s*ada\s*(gunanya|artinya|tujuan)\b', re.IGNORECASE),
        re.compile(r'\b(gaada|ga\s*ada)\s*(harapan|masa\s*depan)\b', re.IGNORECASE),
        re.compile(r'\blebih\s*baik\s*(aku|saya|gue)\s*(pergi|hilang|mati)\b', re.IGNORECASE),
        re.compile(r'\bselesai(kan)?\s*(hidup|semua)\b', re.IGNORECASE),
        re.compile(r'\b(nyerah|menyerah)\b', re.IGNORECASE),
        re.compile(r'\bgak?\s*berarti\b', re.IGNORECASE),
        # Self-harm
        re.compile(r'\b(nyakitin|menyakiti|melukai)\s*diri\b', re.IGNORECASE),
        re.compile(r'\b(iris|sayat|potong)\s*(tangan|nadi|pergelangan)\b', re.IGNORECASE),
    ]

    # Minimum turns required before a stage can transition
    MIN_TURNS = {
        'pembukaan': 1,
        'pembahasan': 1,
        'intervensi': 1,
        'solusi': 1,
        'relaksasi': 1,
        'penutupan': 1,
    }

    # Maximum turns — force transition after this many turns (safety net)
    MAX_TURNS = {
        'pembukaan': 2,
        'pembahasan': 4,
        'intervensi': 3,
        'solusi': 3,
        'relaksasi': 3,
        'penutupan': 99,  # never force-exit penutupan
    }

    # Transition signal patterns per stage (Indonesian phrases)
    # After minimum turns, if any of these patterns are found in user's message,
    # the session advances to the next stage.
    TRANSITION_SIGNALS = {
        'pembukaan': [
            # Any response from user triggers transition from pembukaan
            # We handle this via MIN_TURNS = 1 (auto-advance)
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
            # Transition signals for accepting the professional referral offer
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

    # Pre-compiled regex patterns for fast matching (compiled once at class load)
    _COMPILED_SIGNALS = {
        stage: [re.compile(p) for p in patterns]
        for stage, patterns in TRANSITION_SIGNALS.items()
    }

    def __init__(self, rag_engine):
        """
        Initialize the session manager.

        Args:
            rag_engine: An initialized RAGEngine instance.
        """
        self.rag_engine = rag_engine
        self.current_stage = 'pembukaan'
        self.stage_index = 0
        self.turn_count = 0
        self.conversation_history = []  # list of (role, text) tuples
        self.accumulated_intents = Counter()  # intent → frequency
        self.primary_intents = []  # top intents derived from pembahasan
        self.session_ended = False
        self.has_physical_symptoms = False  # tracks Mengisyaratkan Gejala Fisik across session
        self.in_professional_stage = False  # True when in bantuan_profesional branch
        self.used_verse_books = set()  # book_abbr of books already given as verse in this session
        self.used_verses = set()  # exact reference strings (e.g. "Mazmur 34:18") already given in this session

    def chat(self, user_input):
        """
        Main entry point for a conversation turn.
        Handles state management, calls RAG engine, and manages transitions.

        Args:
            user_input: The user's message text.

        Returns:
            dict with keys:
                - response: The chatbot's response text
                - debug: Debug info dict (stage, turn_count, transitioned, accumulated_intents, etc.)
                - show_professional_button: True when bantuan_profesional stage is active
        """
        if self.session_ended:
            return {
                "response": "Sesi konseling sudah berakhir. Semoga kamu merasa lebih baik. Tuhan memberkati!",
                "debug": {"stage": "ended", "turn_count": self.turn_count},
                "show_professional_button": False
            }

        # Record user message in history
        self.conversation_history.append(("user", user_input))
        self.turn_count += 1

        # Determine which intents to use for bible verse retrieval
        # In relaksasi/solusi, use accumulated intents from the whole session
        override_intents = None
        if self.current_stage in ['relaksasi', 'solusi'] and self.primary_intents:
            override_intents = self.primary_intents

        # Generate response from RAG engine
        result = self.rag_engine.generate_response(
            user_input,
            current_stage=self.current_stage,
            override_intents=override_intents,
            has_physical_symptoms=self.has_physical_symptoms,
            excluded_books=self.used_verse_books,
            excluded_verses=self.used_verses
        )

        # Track which book and exact verse were used (session-level exclusion)
        chosen_book = result['context_used'].get('bible_book_abbr', '')
        chosen_ref = result['context_used'].get('bible_reference', '')
        if chosen_book:
            self.used_verse_books.add(chosen_book)
            print(f"[Session] Verse book '{chosen_book}' added to exclusion set: {self.used_verse_books}")
        if chosen_ref:
            self.used_verses.add(chosen_ref)
            print(f"[Session] Verse '{chosen_ref}' added to verse exclusion set: {self.used_verses}")

        # Accumulate intents from this turn
        turn_intents = result['context_used']['intents']
        for intent in turn_intents:
            self.accumulated_intents[intent] += 1

        # Track physical symptom intent across the entire session
        if self.PHYSICAL_SYMPTOM_INTENT in turn_intents:
            self.has_physical_symptoms = True

        # Update primary intents (top intents by frequency)
        self._update_primary_intents()

        # Record chatbot response in history
        self.conversation_history.append(("counselor", result['response']))

        # --- Emergency skip: keyword-based safety net ---
        # Check for suicidal/hopeless keywords BEFORE relying on the classifier.
        # This catches cases the model misses (e.g., informal/slang expressions).
        keyword_triggered = self._detect_professional_keywords(user_input)

        # --- Emergency skip: classifier OR keyword based ---
        if (self.PROFESSIONAL_INTENT in turn_intents or keyword_triggered) and not self.in_professional_stage:
            return self._enter_professional_stage(user_input)

        # Check for transition
        transitioned = False
        previous_stage = self.current_stage

        if self._should_transition(user_input):
            # Penutupan → bantuan_profesional (user accepted the referral offer)
            if self.current_stage == 'penutupan':
                return self._enter_professional_stage(user_input)
            else:
                self._advance_stage()
                transitioned = True

        # Build debug info
        debug = {
            "stage": previous_stage,
            "new_stage": self.current_stage if transitioned else previous_stage,
            "turn_count": self.turn_count,
            "transitioned": transitioned,
            "accumulated_intents": dict(self.accumulated_intents),
            "primary_intents": self.primary_intents,
            "intents_this_turn": turn_intents,
            "bible_verses": result['context_used']['bible_verses'],
            "example_answer": result['context_used']['example_answer'],
            "has_physical_symptoms": self.has_physical_symptoms,
        }

        return {
            "response": result['response'],
            "debug": debug,
            "show_professional_button": False
        }

    def force_advance(self):
        """
        Manually force-advance to the next stage (for development/testing).

        Returns:
            str: The new stage name, or None if already at the last stage.
        """
        if self.stage_index < len(self.STAGE_ORDER) - 1:
            self._advance_stage()
            return self.current_stage
        return None

    def get_state(self):
        """Return current session state for inspection."""
        return {
            "stage": self.current_stage,
            "stage_index": self.stage_index,
            "turn_count": self.turn_count,
            "total_messages": len(self.conversation_history),
            "accumulated_intents": dict(self.accumulated_intents),
            "primary_intents": self.primary_intents,
            "session_ended": self.session_ended,
        }

    # ── Private methods ──────────────────────────────────────────────

    def _should_transition(self, user_input):
        """
        Determine if the session should transition to the next stage.
        Uses a combination of:
        1. Minimum turn count threshold
        2. Transition signal detection in user's message
        3. Maximum turn count safety net
        """
        # Already at the last stage — no transition
        if self.stage_index >= len(self.STAGE_ORDER) - 1:
            return False

        stage = self.current_stage

        # Pembukaan: always transition after the user's first response
        if stage == 'pembukaan' and self.turn_count >= self.MIN_TURNS[stage]:
            return True

        # Maximum turns exceeded — force transition
        if self.turn_count >= self.MAX_TURNS.get(stage, 99):
            return True

        # Below minimum turns — never transition
        if self.turn_count < self.MIN_TURNS.get(stage, 1):
            return False

        # Check for transition signals in user's message
        return self._detect_transition_signal(user_input, stage)

    def _detect_transition_signal(self, user_input, stage):
        """
        Check if the user's message contains transition cues for the given stage.
        Uses pre-compiled regex patterns for fast matching.
        """
        compiled = self._COMPILED_SIGNALS.get(stage, [])
        if not compiled:
            return False

        text_lower = user_input.lower().strip()

        for pattern in compiled:
            if pattern.search(text_lower):
                return True

        return False

    def _advance_stage(self):
        """Move to the next stage and reset turn count."""
        if self.stage_index < len(self.STAGE_ORDER) - 1:
            self.stage_index += 1
            self.current_stage = self.STAGE_ORDER[self.stage_index]
            self.turn_count = 0

            # When advancing past pembahasan, snapshot the primary intents
            if self.current_stage == 'intervensi':
                self._snapshot_primary_intents()

    def _enter_professional_stage(self, user_input):
        """
        Transition to the bantuan_profesional branch stage.
        Generates a response with a Bible verse about seeking help,
        sets session_ended, and signals the frontend to show the button.
        """
        self.current_stage = self.PROFESSIONAL_STAGE
        self.in_professional_stage = True
        self.turn_count = 0

        # Override intents with a meaningful query so FAISS retrieves an
        # encouraging Bible verse about hope, seeking help, and God's support.
        # This is hardcoded because the bantuan_profesional stage always needs
        # the same type of verse regardless of the user's specific words.
        override = ['pertolongan harapan kekuatan Tuhan membantu tidak sendirian']

        result = self.rag_engine.generate_response(
            user_input,
            current_stage=self.PROFESSIONAL_STAGE,
            override_intents=override,
            has_physical_symptoms=self.has_physical_symptoms,
            excluded_books=self.used_verse_books,
            excluded_verses=self.used_verses
        )

        # Track which book and exact verse were used (session-level exclusion)
        chosen_book = result['context_used'].get('bible_book_abbr', '')
        chosen_ref = result['context_used'].get('bible_reference', '')
        if chosen_book:
            self.used_verse_books.add(chosen_book)
            print(f"[Session] Verse book '{chosen_book}' added to exclusion set: {self.used_verse_books}")
        if chosen_ref:
            self.used_verses.add(chosen_ref)
            print(f"[Session] Verse '{chosen_ref}' added to verse exclusion set: {self.used_verses}")

        self.session_ended = True
        self.conversation_history.append(("counselor", result['response']))

        debug = {
            "stage": self.PROFESSIONAL_STAGE,
            "new_stage": self.PROFESSIONAL_STAGE,
            "turn_count": self.turn_count,
            "transitioned": True,
            "accumulated_intents": dict(self.accumulated_intents),
            "primary_intents": self.primary_intents,
            "intents_this_turn": result['context_used']['intents'],
            "bible_verses": result['context_used']['bible_verses'],
            "example_answer": result['context_used']['example_answer'],
            "has_physical_symptoms": self.has_physical_symptoms,
        }

        return {
            "response": result['response'],
            "debug": debug,
            "show_professional_button": True
        }

    def _detect_professional_keywords(self, user_input):
        """
        Check if the user's message contains suicidal/hopeless keywords.
        Acts as a safety net alongside the classifier for critical cases.

        Returns:
            bool: True if any professional help keyword pattern matches.
        """
        for pattern in self.PROFESSIONAL_KEYWORDS:
            if pattern.search(user_input):
                return True
        return False

    def _update_primary_intents(self):
        """Update the primary intents list based on accumulated frequency."""
        if self.accumulated_intents:
            # Sort by frequency (descending), take top 3
            sorted_intents = self.accumulated_intents.most_common(3)
            self.primary_intents = [intent for intent, _ in sorted_intents]

    def _snapshot_primary_intents(self):
        """
        Take a snapshot of primary intents when transitioning out of pembahasan.
        These will be used for bible verse retrieval in later stages.
        """
        self._update_primary_intents()

    def reset(self):
        """Reset the session to start a new counseling session."""
        self.current_stage = 'pembukaan'
        self.stage_index = 0
        self.turn_count = 0
        self.conversation_history = []
        self.accumulated_intents = Counter()
        self.primary_intents = []
        self.session_ended = False
        self.has_physical_symptoms = False
        self.in_professional_stage = False
        self.used_verse_books = set()
        self.used_verses = set()