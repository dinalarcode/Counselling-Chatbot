"""Self-check for the consent gateways and paced solution delivery in the SessionManager.

Run from the repo root: python -m core.session_manager.test_consent_gates
"""

from core.session_manager.SM_Engine import SessionManager


class StubRAGEngine:
    """Records every generate_response call and returns a canned result without any model or network."""

    def __init__(self):
        self.calls = []  # List of kwargs dicts, one per generate_response call.

    def generate_response(self, user_input, **kwargs):
        self.calls.append(kwargs)
        return {
            "response": "respons konselor",
            "context_used": {
                "intents": [],
                "example_answer": "",
                "bible_verses": "",
                "bible_reference": "",
                "bible_book_abbr": "",
                "stage": kwargs.get("current_stage", ""),
            },
        }

    def generate_background_summary(self, history):
        return "ringkasan keluhan"

    def generate_technique_extraction(self, history):
        return "Pernapasan 4-7-8"


def drive_to_intervensi(sm):
    """Walk a session from pembukaan to intervensi using neutral, signal-free pembahasan messages."""
    sm.chat("halo")  # pembukaan auto-advances.
    assert sm.current_stage == "pembahasan", sm.current_stage
    sm.chat("aku merasa sedih karena masalah pekerjaan")
    sm.chat("atasanku sering memarahiku tanpa alasan")
    sm.chat("aku merasa tertekan setiap hari")
    sm.chat("cuma itu saja ceritaku")  # Early-exit signal advances to intervensi.
    assert sm.current_stage == "intervensi", sm.current_stage


def test_decline_relaxation_skips_to_penutupan_with_recap():
    stub = StubRAGEngine()
    sm = SessionManager(stub)
    drive_to_intervensi(sm)

    # Intervensi transition signal fires the spiritual consent gate instead of transitioning.
    sm.chat("iya benar, masuk akal")
    assert sm.current_stage == "intervensi"
    assert sm.spiritual_consent_asked is True
    assert stub.calls[-1]["ask_spiritual_consent"] is True

    # The answer turn captures consent and forces the transition to solusi.
    sm.chat("iya, aku bersedia")
    assert sm.spiritual_consent is True
    assert sm.current_stage == "solusi"

    # Solusi turn 1 is practical only: no verse yet even though consent is given.
    sm.chat("hmm, bagaimana caranya?")
    assert sm.current_stage == "solusi"
    assert stub.calls[-1]["turn_in_stage"] == 1

    # Turn 2 is the spiritual turn; a transition signal must NOT exit before the verse is delivered.
    sm.chat("terima kasih, sarannya membantu")
    assert sm.current_stage == "solusi"
    assert stub.calls[-1]["turn_in_stage"] == 2
    assert stub.calls[-1]["ask_relaxation_consent"] is False

    # Turn 3 hits MAX_TURNS so the relaxation consent gate fires and holds the stage.
    sm.chat("baik, aku mengerti")
    assert sm.current_stage == "solusi"
    assert sm.relaxation_consent_asked is True
    assert stub.calls[-1]["ask_relaxation_consent"] is True

    # Declining relaxation skips relaksasi entirely and lands on penutupan.
    sm.chat("tidak mau")
    assert sm.relaxation_consent is False
    assert sm.current_stage == "penutupan"
    assert sm.relaxation_skipped is True

    # The first penutupan response opens with the session recap fed from the solusi history.
    sm.chat("hmm")
    assert stub.calls[-1]["closing_recap"] is True
    assert stub.calls[-1]["solusi_history"], "solusi transcript must feed the recap"


def test_accept_relaxation_reaches_relaksasi():
    stub = StubRAGEngine()
    sm = SessionManager(stub)
    drive_to_intervensi(sm)

    sm.chat("saya paham")  # Gate: spiritual consent question.
    sm.chat("tidak usah pakai Alkitab")  # Decline spiritual consent.
    assert sm.spiritual_consent is False
    assert sm.current_stage == "solusi"

    # Without spiritual consent every solusi turn stays practical until MAX_TURNS fires the relaxation gate.
    sm.chat("hmm bagaimana menurutmu?")
    sm.chat("lalu apa lagi yang harus kulakukan?")
    sm.chat("hmm begitu rupanya")  # Turn 3 = MAX_TURNS, relaxation gate fires.
    assert sm.current_stage == "solusi"
    assert stub.calls[-1]["ask_relaxation_consent"] is True

    # Accepting relaxation proceeds to relaksasi and the technique extraction hook runs.
    sm.chat("iya mau, ayo dicoba")
    assert sm.relaxation_consent is True
    assert sm.current_stage == "relaksasi"
    assert sm.relaxation_skipped is False
    assert sm.chosen_technique == "Pernapasan 4-7-8"


if __name__ == "__main__":
    test_decline_relaxation_skips_to_penutupan_with_recap()
    test_accept_relaxation_reaches_relaksasi()
    print("OK - consent gateway self-checks passed")
