"""Session manager engine that drives automatic stage progression for the counseling chatbot."""

from collections import Counter

from core.session_manager.Listof_List import (
    STAGE_ORDER, PROFESSIONAL_STAGE, PROFESSIONAL_INTENT, PHYSICAL_SYMPTOM_INTENT,
    PROFESSIONAL_KEYWORDS, DECLINE_SIGNALS, SPIRITUAL_CONSENT_DECLINE,
    SPIRITUAL_CONSENT_AFFIRM, MIN_TURNS, MAX_TURNS,
    PEMBAHASAN_EARLY_EXIT_SIGNALS, TRANSITION_SIGNALS, _COMPILED_SIGNALS,
)


class SessionManager:
    """Wraps the RAG engine and manages counseling session state with automatic stage transitions."""

    # Static data imported from Listof_List and bound as class attributes so self access keeps working.
    STAGE_ORDER = STAGE_ORDER
    PROFESSIONAL_STAGE = PROFESSIONAL_STAGE
    PROFESSIONAL_INTENT = PROFESSIONAL_INTENT
    PHYSICAL_SYMPTOM_INTENT = PHYSICAL_SYMPTOM_INTENT
    PROFESSIONAL_KEYWORDS = PROFESSIONAL_KEYWORDS
    DECLINE_SIGNALS = DECLINE_SIGNALS
    SPIRITUAL_CONSENT_DECLINE = SPIRITUAL_CONSENT_DECLINE
    SPIRITUAL_CONSENT_AFFIRM = SPIRITUAL_CONSENT_AFFIRM
    MIN_TURNS = MIN_TURNS
    MAX_TURNS = MAX_TURNS
    PEMBAHASAN_EARLY_EXIT_SIGNALS = PEMBAHASAN_EARLY_EXIT_SIGNALS
    TRANSITION_SIGNALS = TRANSITION_SIGNALS
    _COMPILED_SIGNALS = _COMPILED_SIGNALS

    def __init__(self, rag_engine):
        """Initialize the session manager with an initialized RAGEngine instance."""
        self.rag_engine = rag_engine
        self.current_stage = 'pembukaan'
        self.stage_index = 0
        self.turn_count = 0
        self.turn_in_stage = 0  # Per-stage turn counter that resets on each stage transition.
        self.conversation_history = []  # List of role and text tuples.
        self.accumulated_intents = Counter()  # Maps intent to frequency.
        self.primary_intents = []  # Top intents derived from pembahasan.
        self.session_ended = False
        self.has_physical_symptoms = False  # Tracks the physical symptom intent across the session.
        self.in_professional_stage = False  # True when in the bantuan_profesional branch.
        self.used_verse_books = set()  # Book abbreviations already given as a verse this session.
        self.used_verses = set()  # Exact reference strings already given this session.
        # Tri-state spiritual consent gating all Bible verse injection in later stages.
        self.spiritual_consent = None
        self.spiritual_consent_asked = False  # One-shot flag so the consent question is shown once.
        # Ephemeral pembahasan history cleared right after summary generation at the pembahasan to intervensi transition.
        self.temp_pembahasan_history = []   # List of user message and bot response tuples.
        self.complaint_summary = None       # One-sentence summary injected into later stages.
        # Ephemeral solusi history cleared right after technique extraction at the solusi to relaksasi transition.
        self.temp_solusi_history = []       # List of user message and bot response tuples.
        self.chosen_technique = None        # Technique name injected into the relaksasi prompt.

    def chat(self, user_input):
        """Main entry point for a conversation turn handling state, RAG generation, and transitions."""
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

        # Record user message in history.
        self.conversation_history.append(("user", user_input))
        self.turn_count += 1
        self.turn_in_stage += 1

        # Capture the sticky reply to the spiritual consent question so a clear yes or no is honored even without a transition.
        if self.spiritual_consent_asked and self.spiritual_consent is None:
            answer = self._detect_spiritual_consent(user_input)
            if answer is not None:
                self.spiritual_consent = answer

        # Ask spiritual consent on the final solusi turn so the yes or no question stays separate from exploration.
        ask_consent = (
            self.current_stage == 'solusi'
            and self.turn_in_stage >= self.MAX_TURNS.get('solusi', 3)
            and not self.spiritual_consent_asked
        )

        # Only relaksasi retrieves verses, so pass accumulated primary intents there for best results.
        override_intents = None
        if self.current_stage == 'relaksasi' and self.primary_intents:
            override_intents = self.primary_intents

        # Generate response from RAG engine.
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

        # Track which book and exact verse were used for session-level exclusion.
        chosen_book = result['context_used'].get('bible_book_abbr', '')
        chosen_ref = result['context_used'].get('bible_reference', '')
        if chosen_book:
            self.used_verse_books.add(chosen_book)
            print(f"[Session] Verse book '{chosen_book}' added to exclusion set: {self.used_verse_books}")
        if chosen_ref:
            self.used_verses.add(chosen_ref)
            print(f"[Session] Verse '{chosen_ref}' added to verse exclusion set: {self.used_verses}")

        # Accumulate intents from this turn.
        turn_intents = result['context_used']['intents']
        for intent in turn_intents:
            self.accumulated_intents[intent] += 1

        # Track the physical symptom intent across the entire session.
        if self.PHYSICAL_SYMPTOM_INTENT in turn_intents:
            self.has_physical_symptoms = True

        # Update primary intents by frequency.
        self._update_primary_intents()

        # Record chatbot response in history.
        self.conversation_history.append(("counselor", result['response']))

        # Record pembahasan turns for background summarization at the stage transition.
        self._record_pembahasan_turn(user_input, result['response'])

        # Record solusi turns for technique extraction at the solusi to relaksasi transition.
        self._record_solusi_turn(user_input, result['response'])

        # Emergency skip keyword safety net that catches cases the classifier misses.
        keyword_triggered = self._detect_professional_keywords(user_input)

        # The keyword safety net fires at any stage while classifier detection is restricted to early stages to avoid hijacking recovery.
        EARLY_STAGES = {'pembukaan', 'pembahasan', 'intervensi'}
        classifier_positive = self.PROFESSIONAL_INTENT in turn_intents
        should_skip_to_professional = (
            keyword_triggered
            or (classifier_positive and self.current_stage in EARLY_STAGES)
        ) and not self.in_professional_stage

        if should_skip_to_professional:
            return self._enter_professional_stage(user_input)

        # Check for transition.
        transitioned = False
        previous_stage = self.current_stage

        if self._should_transition(user_input):
            # Penutupan to bantuan_profesional when the user accepts the referral offer.
            if self.current_stage == 'penutupan':
                return self._enter_professional_stage(user_input)
            else:
                self._advance_stage()
                transitioned = True

        # Decline path at penutupan closes the session gracefully with a warm farewell.
        elif self.current_stage == 'penutupan' and self._detect_decline_signal(user_input):
            return self._end_session(user_input)

        # Build debug info.
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
        """Manually force-advance to the next stage, returning the new stage or None at the last stage."""
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

    # Private methods.

    def _should_transition(self, user_input):
        """Decide whether to transition using minimum turns, transition signals, maximum turns, and pembahasan early-exit."""
        # Already at the last stage so no transition.
        if self.stage_index >= len(self.STAGE_ORDER) - 1:
            return False

        stage = self.current_stage

        # Pembukaan always transitions after the user's first response.
        if stage == 'pembukaan' and self.turn_count >= self.MIN_TURNS[stage]:
            return True

        # Maximum turns exceeded forces a transition.
        if self.turn_count >= self.MAX_TURNS.get(stage, 99):
            return True

        # Pembahasan early-exit allows immediate transition regardless of turn count to respect user autonomy.
        if stage == 'pembahasan' and self._detect_pembahasan_early_exit(user_input):
            print(f"[Session] pembahasan early-exit triggered at turn_in_stage={self.turn_in_stage}")
            return True

        # Below minimum turns never transitions.
        if self.turn_count < self.MIN_TURNS.get(stage, 1):
            return False

        # Check for transition signals in the user's message.
        return self._detect_transition_signal(user_input, stage)

    def _detect_transition_signal(self, user_input, stage):
        """Check if the user's message contains transition cues for the stage using pre-compiled patterns."""
        compiled = self._COMPILED_SIGNALS.get(stage, [])
        if not compiled:
            return False

        text_lower = user_input.lower().strip()

        for pattern in compiled:
            if pattern.search(text_lower):
                return True

        return False

    def _advance_stage(self):
        """Move to the next stage and reset turn counters, running pembahasan and solusi transition hooks."""
        if self.stage_index < len(self.STAGE_ORDER) - 1:
            self.stage_index += 1
            self.current_stage = self.STAGE_ORDER[self.stage_index]
            self.turn_count = 0
            self.turn_in_stage = 0  # Reset per-stage counter on transition.

            # When advancing to intervensi, snapshot intents and generate a one-sentence complaint summary for later prompts.
            if self.current_stage == 'intervensi':
                self._snapshot_primary_intents()
                if self.temp_pembahasan_history:
                    self.complaint_summary = self.rag_engine.generate_background_summary(
                        self.temp_pembahasan_history
                    )
                    self.temp_pembahasan_history = []  # Immediately free ephemeral data.

            # When advancing to relaksasi, extract the chosen relaxation technique from the solusi history for the prompt.
            if self.current_stage == 'relaksasi':
                if self.temp_solusi_history:
                    self.chosen_technique = self.rag_engine.generate_technique_extraction(
                        self.temp_solusi_history
                    )
                    self.temp_solusi_history = []  # Immediately free ephemeral data.

    def _enter_professional_stage(self, user_input):
        """Transition to the bantuan_profesional branch with an encouraging verse, ending the session and signaling the button."""
        self.current_stage = self.PROFESSIONAL_STAGE
        self.in_professional_stage = True
        self.turn_count = 0

        # Hardcoded override so FAISS retrieves an encouraging verse about hope and seeking help regardless of the user's words.
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

        # Track which book and exact verse were used for session-level exclusion.
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
        """Return True if the user's message matches any suicidal or hopeless keyword pattern."""
        for pattern in self.PROFESSIONAL_KEYWORDS:
            if pattern.search(user_input):
                return True
        return False

    def _detect_pembahasan_early_exit(self, user_input):
        """Return True if the user signals during pembahasan that they have nothing more to share."""
        text_lower = user_input.lower().strip()
        for pattern in self.PEMBAHASAN_EARLY_EXIT_SIGNALS:
            if pattern.search(text_lower):
                return True
        return False

    def _detect_spiritual_consent(self, user_input):
        """Interpret the user's reply to the spiritual consent question as True, False, or None when unclear."""
        text_lower = user_input.lower().strip()

        for pattern in self.SPIRITUAL_CONSENT_DECLINE:
            if pattern.search(text_lower):
                return False

        for pattern in self.SPIRITUAL_CONSENT_AFFIRM:
            if pattern.search(text_lower):
                return True

        return None

    def _detect_decline_signal(self, user_input):
        """Return True if the user declines the professional referral at penutupan."""
        text_lower = user_input.lower().strip()
        for pattern in self.DECLINE_SIGNALS:
            if pattern.search(text_lower):
                return True
        return False

    def _end_session(self, user_input):
        """Close the session gracefully with a warm farewell after the user declines the penutupan referral."""
        # Use the dedicated closing instruction so the LLM gives a warm farewell without repeating the referral offer.
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
        """Update the primary intents list to the top three accumulated intents by frequency."""
        if self.accumulated_intents:
            # Sort by frequency descending and take the top three.
            sorted_intents = self.accumulated_intents.most_common(3)
            self.primary_intents = [intent for intent, _ in sorted_intents]

    def _record_pembahasan_turn(self, user_input, bot_response):
        """Append a turn to the ephemeral pembahasan buffer, only during the pembahasan stage."""
        if self.current_stage == 'pembahasan':
            self.temp_pembahasan_history.append((user_input, bot_response))

    def _record_solusi_turn(self, user_input, bot_response):
        """Append a turn to the ephemeral solusi buffer, only during the solusi stage."""
        if self.current_stage == 'solusi':
            self.temp_solusi_history.append((user_input, bot_response))

    def _snapshot_primary_intents(self):
        """Snapshot primary intents when transitioning out of pembahasan for later verse retrieval."""
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
