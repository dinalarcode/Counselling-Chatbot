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
        re.compile(r'\b(udah|sudah|aku|saya|gue|gw)\s+(nyerah|menyerah)\b', re.IGNORECASE),
        re.compile(r'\bgak?\s*berarti\b', re.IGNORECASE),
        # Self-harm
        re.compile(r'\b(nyakitin|menyakiti|melukai)\s*diri\b', re.IGNORECASE),
        re.compile(r'\b(iris|sayat|potong)\s*(tangan|nadi|pergelangan)\b', re.IGNORECASE),
    ]

    # Decline signals for penutupan — user declines professional referral offer.
    # Only checked at penutupan stage. When matched, session ends gracefully.
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

    # Spiritual-consent detection patterns (reply to the consent question asked
    # at the intervensi stage). Decline is checked first so phrases like
    # "tidak mau" resolve to a refusal rather than matching the affirmative "mau".
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

    # Minimum turns required before a stage can transition
    MIN_TURNS = {
        'pembukaan': 1,
        'pembahasan': 3,  # clinical requirement: minimum 3 exploration turns
        'intervensi': 1,
        'solusi': 1,
        'relaksasi': 1,
        'penutupan': 1,
    }

    # Maximum turns — force transition after this many turns (safety net)
    MAX_TURNS = {
        'pembukaan': 2,
        'pembahasan': 5,  # extended ceiling to allow richer exploration
        'intervensi': 3,
        'solusi': 3,
        'relaksasi': 3,
        'penutupan': 99,  # never force-exit penutupan
    }

    # Early-exit signals for pembahasan — when the user explicitly indicates
    # they have nothing more to share, we bypass the MIN_TURNS requirement
    # and allow an immediate transition to the intervensi stage.
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
            # NOTE: r'\bbisa\b' was removed — too broad for Indonesian; causes
            # false transitions on common phrases like "bisa jadi", "bisa kan", etc.
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
        self.turn_in_stage = 0  # per-stage turn counter; resets on each stage transition
        self.conversation_history = []  # list of (role, text) tuples
        self.accumulated_intents = Counter()  # intent → frequency
        self.primary_intents = []  # top intents derived from pembahasan
        self.session_ended = False
        self.has_physical_symptoms = False  # tracks Mengisyaratkan Gejala Fisik across session
        self.in_professional_stage = False  # True when in bantuan_profesional branch
        self.used_verse_books = set()  # book_abbr of books already given as verse in this session
        self.used_verses = set()  # exact reference strings (e.g. "Mazmur 34:18") already given in this session
        # Tri-state spiritual consent: None = not yet answered, True = consented,
        # False = declined. Gates all Bible verse injection in later stages.
        self.spiritual_consent = None
        self.spiritual_consent_asked = False  # one-shot: consent question shown once
        # Ephemeral pembahasan history for background summarization.
        # Cleared immediately after summary generation at pembahasan→intervensi transition.
        self.temp_pembahasan_history = []   # list of (user_msg, bot_response) tuples
        self.complaint_summary = None       # 1-sentence summary injected into later stages
        # Ephemeral solusi history for relaxation technique extraction.
        # Cleared immediately after extraction at solusi→relaksasi transition.
        self.temp_solusi_history = []       # list of (user_msg, bot_response) tuples
        self.chosen_technique = None        # technique name injected into relaksasi prompt

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
                "response": (
                    "Sesi konseling kita sudah selesai. 🙏\n\n"
                    "Terima kasih sudah mau berbagi dan terbuka hari ini. "
                    "Ingatlah bahwa kamu tidak sendirian — Tuhan selalu menyertaimu. "
                    "Jika kamu ingin bercerita lagi, kamu bisa memulai sesi baru kapan saja. "
                    "Semoga harimu penuh damai dan berkat. 🌿"
                ),
                "debug": {"stage": "ended", "turn_count": self.turn_count},
                "show_professional_button": False,
                "session_ended": True
            }

        # Record user message in history
        self.conversation_history.append(("user", user_input))
        self.turn_count += 1
        self.turn_in_stage += 1

        # Capture the user's reply to the spiritual-consent question (asked at final
        # solusi turn). Sticky: first clear yes/no wins and is never overwritten.
        # Checked here so "tidak" replies (not a transition signal) are still honored
        # even if they don't trigger a stage transition.
        if self.spiritual_consent_asked and self.spiritual_consent is None:
            answer = self._detect_spiritual_consent(user_input)
            if answer is not None:
                self.spiritual_consent = answer

        # Ask spiritual consent on the FINAL turn of solusi (turn_in_stage == MAX).
        # Asking at solusi end (not intervensi) avoids mixing the yes/no question
        # with exploration questions, and consent is resolved before relaksasi begins.
        ask_consent = (
            self.current_stage == 'solusi'
            and self.turn_in_stage >= self.MAX_TURNS.get('solusi', 3)
            and not self.spiritual_consent_asked
        )

        # Determine which intents to use for bible verse retrieval.
        # Only relaksasi now retrieves verses; pass accumulated primary intents for best results.
        override_intents = None
        if self.current_stage == 'relaksasi' and self.primary_intents:
            override_intents = self.primary_intents

        # Generate response from RAG engine
        result = self.rag_engine.generate_response(
            user_input,
            current_stage=self.current_stage,
            override_intents=override_intents,
            has_physical_symptoms=self.has_physical_symptoms,
            excluded_books=self.used_verse_books,
            excluded_verses=self.used_verses,
            spiritual_consent=self.spiritual_consent,
            ask_spiritual_consent=ask_consent,
            turn_in_stage=self.turn_in_stage,
            complaint_summary=self.complaint_summary,
            chosen_technique=self.chosen_technique
        )

        # Mark the consent question as shown so it is not repeated.
        if ask_consent:
            self.spiritual_consent_asked = True

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

        # Record pembahasan turns for background summarization at stage transition.
        # Only collected during pembahasan; list is cleared after summary is generated.
        self._record_pembahasan_turn(user_input, result['response'])

        # Record solusi turns for technique extraction at the solusi→relaksasi transition.
        # Only collected during solusi; list is cleared after extraction.
        self._record_solusi_turn(user_input, result['response'])

        # --- Emergency skip: keyword-based safety net ---
        # Check for suicidal/hopeless keywords BEFORE relying on the classifier.
        # This catches cases the model misses (e.g., informal/slang expressions).
        keyword_triggered = self._detect_professional_keywords(user_input)

        # --- Emergency skip: classifier + keyword based (stage-gated) ---
        # Keyword safety net (suicidal/self-harm) fires at ANY stage.
        # Classifier-based detection is restricted to early stages only to
        # prevent false positives from hijacking the therapeutic flow in
        # later stages (solusi/relaksasi) where the user is already in recovery.
        EARLY_STAGES = {'pembukaan', 'pembahasan', 'intervensi'}
        classifier_positive = self.PROFESSIONAL_INTENT in turn_intents
        should_skip_to_professional = (
            keyword_triggered
            or (classifier_positive and self.current_stage in EARLY_STAGES)
        ) and not self.in_professional_stage

        if should_skip_to_professional:
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

        # Decline path for penutupan — user declines professional referral,
        # so we close the session gracefully with a warm farewell.
        elif self.current_stage == 'penutupan' and self._detect_decline_signal(user_input):
            return self._end_session(user_input)

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
        4. Pembahasan early-exit: explicit "nothing more to share" phrases bypass
           the 3-turn minimum so the user is never trapped in the exploration loop.
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

        # Pembahasan early-exit: if the user explicitly says they have nothing
        # more to share, we allow immediate transition regardless of turn count.
        # This overrides the 3-turn minimum to respect user autonomy.
        if stage == 'pembahasan' and self._detect_pembahasan_early_exit(user_input):
            print(f"[Session] pembahasan early-exit triggered at turn_in_stage={self.turn_in_stage}")
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
            self.turn_in_stage = 0  # reset per-stage counter on transition

            # When advancing past pembahasan, snapshot intents and generate
            # a one-sentence background summary of the client's core complaint.
            # The summary is injected into intervensi/solusi/relaksasi prompts
            # so the stateless LLM retains context across turns.
            if self.current_stage == 'intervensi':
                self._snapshot_primary_intents()
                if self.temp_pembahasan_history:
                    self.complaint_summary = self.rag_engine.generate_background_summary(
                        self.temp_pembahasan_history
                    )
                    self.temp_pembahasan_history = []  # immediately free ephemeral data

            # When advancing to relaksasi, extract the chosen relaxation technique from
            # the solusi conversation history. The technique name is injected into the
            # relaksasi prompt so the LLM guides only that specific technique.
            if self.current_stage == 'relaksasi':
                if self.temp_solusi_history:
                    self.chosen_technique = self.rag_engine.generate_technique_extraction(
                        self.temp_solusi_history
                    )
                    self.temp_solusi_history = []  # immediately free ephemeral data

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
            excluded_verses=self.used_verses,
            spiritual_consent=self.spiritual_consent
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

    def _detect_pembahasan_early_exit(self, user_input):
        """
        Check if the user explicitly signals they have nothing more to share
        during the pembahasan stage. When matched, the 3-turn minimum is
        bypassed and the session transitions immediately to intervensi.

        Returns:
            bool: True if an early-exit phrase is detected.
        """
        text_lower = user_input.lower().strip()
        for pattern in self.PEMBAHASAN_EARLY_EXIT_SIGNALS:
            if pattern.search(text_lower):
                return True
        return False

    def _detect_spiritual_consent(self, user_input):
        """
        Interpret the user's reply to the spiritual-consent question.

        Returns:
            True  — user consents to a biblical perspective (affirmative).
            False — user declines (negative).
            None  — no clear answer detected; caller keeps consent unset and may
                    re-check on a later turn (conservative: no verse without consent).
        """
        text_lower = user_input.lower().strip()

        for pattern in self.SPIRITUAL_CONSENT_DECLINE:
            if pattern.search(text_lower):
                return False

        for pattern in self.SPIRITUAL_CONSENT_AFFIRM:
            if pattern.search(text_lower):
                return True

        return None

    def _detect_decline_signal(self, user_input):
        """
        Check if the user is declining the professional referral at penutupan.
        Only called when current_stage == 'penutupan'.

        Returns:
            bool: True if any decline signal pattern matches.
        """
        text_lower = user_input.lower().strip()
        for pattern in self.DECLINE_SIGNALS:
            if pattern.search(text_lower):
                return True
        return False

    def _end_session(self, user_input):
        """
        Close the session gracefully after the user declines the professional
        referral at penutupan. Generates a warm LLM farewell using a dedicated
        'penutupan_selesai' stage instruction and marks the session as ended.
        """
        # Use the dedicated closing instruction so the LLM gives a final
        # warm farewell without repeating the referral offer
        result = self.rag_engine.generate_response(
            user_input,
            current_stage='penutupan_selesai',
            override_intents=None,
            has_physical_symptoms=self.has_physical_symptoms,
            excluded_books=self.used_verse_books,
            excluded_verses=self.used_verses,
            spiritual_consent=self.spiritual_consent
        )

        self.session_ended = True
        self.conversation_history.append(("counselor", result['response']))

        debug = {
            "stage": "penutupan",
            "new_stage": "ended",
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
            "show_professional_button": False,
            "session_ended": True,
        }

    def _update_primary_intents(self):
        """Update the primary intents list based on accumulated frequency."""
        if self.accumulated_intents:
            # Sort by frequency (descending), take top 3
            sorted_intents = self.accumulated_intents.most_common(3)
            self.primary_intents = [intent for intent, _ in sorted_intents]

    def _record_pembahasan_turn(self, user_input, bot_response):
        """
        Append a turn to the ephemeral pembahasan history buffer.
        Only records during the pembahasan stage; no-op at all other stages.
        The buffer is cleared after the background summary is generated.
        """
        if self.current_stage == 'pembahasan':
            self.temp_pembahasan_history.append((user_input, bot_response))

    def _record_solusi_turn(self, user_input, bot_response):
        """
        Append a turn to the ephemeral solusi history buffer.
        Only records during the solusi stage; no-op at all other stages.
        The buffer is cleared after the technique extraction at the solusi→relaksasi transition.
        """
        if self.current_stage == 'solusi':
            self.temp_solusi_history.append((user_input, bot_response))

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
        self.turn_in_stage = 0
        self.conversation_history = []
        self.accumulated_intents = Counter()
        self.primary_intents = []
        self.session_ended = False
        self.has_physical_symptoms = False
        self.in_professional_stage = False
        self.used_verse_books = set()
        self.used_verses = set()
        self.spiritual_consent = None
        self.spiritual_consent_asked = False
        self.temp_pembahasan_history = []
        self.complaint_summary = None
        self.temp_solusi_history = []
        self.chosen_technique = None