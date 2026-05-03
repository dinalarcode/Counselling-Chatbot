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

    # Ordered counseling stages
    STAGE_ORDER = ['pembukaan', 'pembahasan', 'intervensi', 'solusi', 'relaksasi', 'penutupan']

    # Minimum turns required before a stage can transition
    MIN_TURNS = {
        'pembukaan': 1,
        'pembahasan': 1,
        'intervensi': 1,
        'solusi': 1,
        'relaksasi': 1,
        'penutupan': 1,  # no transition from penutupan
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
        'penutupan': [],  # no transition from penutupan
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
        """
        if self.session_ended:
            return {
                "response": "Sesi konseling sudah berakhir. Semoga kamu merasa lebih baik. Tuhan memberkati!",
                "debug": {"stage": "ended", "turn_count": self.turn_count}
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
            override_intents=override_intents
        )

        # Accumulate intents from this turn
        turn_intents = result['context_used']['intents']
        for intent in turn_intents:
            self.accumulated_intents[intent] += 1

        # Update primary intents (top intents by frequency)
        self._update_primary_intents()

        # Record chatbot response in history
        self.conversation_history.append(("counselor", result['response']))

        # Check for transition
        transitioned = False
        previous_stage = self.current_stage

        if self._should_transition(user_input):
            self._advance_stage()
            transitioned = True

        # Check if we reached penutupan and turn_count > min (session ending)
        if self.current_stage == 'penutupan' and transitioned:
            # Next chat after penutupan response would end the session
            pass

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
        }

        return {
            "response": result['response'],
            "debug": debug
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
        Uses regex pattern matching on Indonesian phrases.
        """
        signals = self.TRANSITION_SIGNALS.get(stage, [])
        if not signals:
            return False

        text_lower = user_input.lower().strip()

        for pattern in signals:
            if re.search(pattern, text_lower):
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