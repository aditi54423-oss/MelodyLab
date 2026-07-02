# rule_based_basic.py

import random


# ------------------------------------------------------------
# Note-to-MIDI mapping
# ------------------------------------------------------------
# MIDI numbers let us measure note distance and build playable notes
# from pitch-class pools such as ["A", "B", "C", "E", "F"].

NOTE_TO_MIDI = {
    "A3": 57,
    "B-3": 58,
    "B3": 59,

    "C4": 60,
    "C#4": 61,
    "D4": 62,
    "E-4": 63,
    "E4": 64,
    "F4": 65,
    "F#4": 66,
    "G4": 67,
    "A-4": 68,
    "A4": 69,
    "B-4": 70,
    "B4": 71,

    "C5": 72,
    "C#5": 73,
    "D5": 74,
    "E-5": 75,
    "E5": 76,
    "F5": 77,
    "F#5": 78,
    "G5": 79,
    "A-5": 80,
    "A5": 81,
    "B-5": 82,
    "B5": 83,
}

MIDI_TO_NOTE = {midi: note for note, midi in NOTE_TO_MIDI.items()}


# ------------------------------------------------------------
# Style settings
# ------------------------------------------------------------
# The rule-based model does not learn transition probabilities from
# the chosen training melody. Instead, the selected melody chooses the
# musical pitch-class pool and home note.
#
# Minuet        -> G major-ish pool
# Amazing Grace -> F major pool
# Sakura Sakura -> A-centered Sakura/In-style pentatonic: A, B, C, E, F

STYLE_SETTINGS = {
    "G": {
        "display_name": "G major",
        "pitch_classes": ["G", "A", "B", "C", "D", "E", "F#"],
        "home_pitch_class": "G",
        "home_note": "G4",
        "min_midi": 60,
        "max_midi": 79,
    },
    "F": {
        "display_name": "F major",
        "pitch_classes": ["F", "G", "A", "B-", "C", "D", "E"],
        "home_pitch_class": "F",
        "home_note": "F4",
        "min_midi": 60,
        "max_midi": 79,
    },
    "C": {
        "display_name": "C major",
        "pitch_classes": ["C", "D", "E", "F", "G", "A", "B"],
        "home_pitch_class": "C",
        "home_note": "C4",
        "min_midi": 60,
        "max_midi": 79,
    },
    "A_SAKURA": {
        "display_name": "A-centered Sakura pentatonic",
        "pitch_classes": ["A", "B", "C", "E", "F"],
        "home_pitch_class": "A",
        "home_note": "A4",
        "min_midi": 57,
        "max_midi": 81,
    },
}

# Mapping from TRAINING_MELODIES keys in melodies.py to rule-based styles.
MELODY_TO_STYLE = {
    "minuet": "G",
    "grace": "F",
    "amazing_grace": "F",
    "amazing grace": "F",
    "sakura": "A_SAKURA",
    "sakura_sakura": "A_SAKURA",
    "sakura sakura": "A_SAKURA",
}


# ------------------------------------------------------------
# Rhythm options
# ------------------------------------------------------------
# 1.0 appears twice, so one-beat notes are slightly more common.

DEFAULT_DURATIONS = [0.5, 1.0, 1.0, 1.5, 2.0]


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def get_pitch_class(note):
    """
    Return the pitch class from an octave-specific note.

    Examples:
    A4   -> A
    B-4  -> B-
    F#5  -> F#
    REST -> REST
    """
    note = str(note).strip()

    if note.upper().startswith("REST"):
        return "REST"

    if len(note) >= 2 and note[1] in ["#", "-"]:
        return note[:2]

    return note[:1]


def pitch_distance(note1, note2):
    """Return absolute pitch distance in semitones."""
    return abs(NOTE_TO_MIDI[note2] - NOTE_TO_MIDI[note1])


def signed_pitch_distance(note1, note2):
    """Return signed pitch distance in semitones."""
    return NOTE_TO_MIDI[note2] - NOTE_TO_MIDI[note1]


def build_candidate_notes(pitch_classes, min_midi=60, max_midi=79):
    """
    Build playable candidate notes from pitch classes and a MIDI range.

    This avoids limiting the model twice. The pitch-class pool decides which
    notes are musical; the MIDI range decides which octaves are playable.
    """
    candidates = []

    for note, midi_value in NOTE_TO_MIDI.items():
        if min_midi <= midi_value <= max_midi:
            if get_pitch_class(note) in pitch_classes:
                candidates.append(note)

    candidates.sort(key=lambda note: NOTE_TO_MIDI[note])
    return candidates


def get_style_for_training_melody(melody_key):
    """Map an app training-melody key to a rule-based style key."""
    normalized = str(melody_key).lower().strip()
    return MELODY_TO_STYLE.get(normalized, "G")


def choose_duration(step_index, length):
    """Choose a duration. The final note is longer for closure."""
    if step_index == length - 1:
        return 2.0

    return random.choice(DEFAULT_DURATIONS)


# ------------------------------------------------------------
# Candidate scoring function
# ------------------------------------------------------------

def score_candidate(
    candidate,
    melody_so_far,
    pitch_classes,
    home_note,
    min_midi,
    max_midi,
    step_index,
    length,
):
    """
    Score one possible next note according to encoded music-theory rules.

    Rules used:
    1. Stay in pitch-class pool
    2. Prefer stepwise motion
    3. Avoid too much repetition
    4. Resolve large leaps
    5. Stay in comfortable MIDI range
    6. Move toward / end on home note
    """
    score = 0
    rules = []
    last_note = melody_so_far[-1]

    # Rule 1: stay in the pitch-class pool.
    if get_pitch_class(candidate) in pitch_classes:
        score += 3
        rules.append("stays in the selected pitch pool")
    else:
        score -= 5
        rules.append("outside the selected pitch pool")

    # Rule 2: prefer stepwise / nearby motion.
    interval = pitch_distance(last_note, candidate)

    if interval <= 2:
        score += 4
        rules.append("uses smooth stepwise motion")
    elif interval <= 5:
        score += 2
        rules.append("uses a moderate melodic movement")
    elif interval <= 7:
        score -= 1
        rules.append("uses a larger leap")
    else:
        score -= 4
        rules.append("large leap penalty")

    # Rule 3: avoid boring repetition.
    if candidate == last_note:
        score -= 2
        rules.append("repeats the previous note")

    if len(melody_so_far) >= 2 and candidate == melody_so_far[-1] == melody_so_far[-2]:
        score -= 5
        rules.append("avoids three repeated notes in a row")

    # Rule 4: resolve a previous large leap by moving stepwise in the opposite direction.
    if len(melody_so_far) >= 2:
        previous_interval = signed_pitch_distance(melody_so_far[-2], melody_so_far[-1])
        next_interval = signed_pitch_distance(melody_so_far[-1], candidate)

        if abs(previous_interval) > 5:
            moves_opposite_direction = previous_interval * next_interval < 0
            moves_stepwise = abs(next_interval) <= 2

            if moves_opposite_direction and moves_stepwise:
                score += 4
                rules.append("resolves the previous leap by moving stepwise in the opposite direction")
            else:
                score -= 3
                rules.append("does not resolve the previous leap clearly")

    # Rule 5: keep the note inside the selected playable MIDI range.
    midi_value = NOTE_TO_MIDI[candidate]

    if min_midi <= midi_value <= max_midi:
        score += 1
        rules.append("keeps the melody in the playable range")
    else:
        score -= 2
        rules.append("moves outside the playable range")

    # Rule 6: prepare and complete the ending on the home note.
    is_final_note = step_index == length - 1

    if is_final_note:
        if candidate == home_note:
            score += 10
            rules.append("ends on the home note")
        else:
            score -= 4
            rules.append("does not end on the home note")
    else:
        remaining_notes = length - step_index - 1

        if remaining_notes <= 3:
            distance_to_home = pitch_distance(candidate, home_note)

            if distance_to_home <= 4:
                score += 2
                rules.append("moves closer to the home note near the ending")

    return score, rules


# ------------------------------------------------------------
# Explanation generator
# ------------------------------------------------------------

def make_explanation(selected):
    """Convert selected note rules into a readable explanation."""
    useful_rules = []

    for rule in selected["rules"]:
        if "penalty" in rule:
            continue
        if "outside" in rule:
            continue
        if "does not" in rule:
            continue
        useful_rules.append(rule)

    if useful_rules:
        return f"{selected['note']} was selected because it " + ", ".join(useful_rules[:3]) + "."

    return f"{selected['note']} was selected because it had the best overall rule score."


# ------------------------------------------------------------
# Main rule-based melody generator
# ------------------------------------------------------------

def generate_rule_based_melody(
    key="G",
    training_melody_key=None,
    length=16,
    pitch_classes=None,
    home_note=None,
    min_midi=None,
    max_midi=None,
    mode="strict",
    return_trace=True,
    seed=None,
):
    """
    Generate a melody using explicit music-theory rules.

    Use either:
    - key="G" / "F" / "C" / "A_SAKURA", or
    - training_melody_key="minuet" / "grace" / "sakura"

    The selected training melody changes the pitch-class pool and home note.
    It does not train transition probabilities.
    """
    if seed is not None:
        random.seed(seed)

    if training_melody_key is not None:
        key = get_style_for_training_melody(training_melody_key)

    settings = STYLE_SETTINGS.get(key, STYLE_SETTINGS["G"])

    if pitch_classes is None:
        pitch_classes = settings["pitch_classes"]

    if home_note is None:
        home_note = settings["home_note"]

    if min_midi is None:
        min_midi = settings["min_midi"]

    if max_midi is None:
        max_midi = settings["max_midi"]

    candidate_notes = build_candidate_notes(
        pitch_classes=pitch_classes,
        min_midi=min_midi,
        max_midi=max_midi,
    )

    if not candidate_notes:
        raise ValueError("No candidate notes were built. Check pitch_classes and MIDI range.")

    if home_note not in candidate_notes:
        candidate_notes.append(home_note)
        candidate_notes.sort(key=lambda note: NOTE_TO_MIDI[note])

    if mode not in ["strict", "creative"]:
        raise ValueError("mode must be either 'strict' or 'creative'")

    melody = [home_note]
    durations = [1.0]
    trace = []

    for step_index in range(1, length):
        scored_candidates = []

        for candidate in candidate_notes:
            score, rules = score_candidate(
                candidate=candidate,
                melody_so_far=melody,
                pitch_classes=pitch_classes,
                home_note=home_note,
                min_midi=min_midi,
                max_midi=max_midi,
                step_index=step_index,
                length=length,
            )

            scored_candidates.append({
                "note": candidate,
                "score": round(score, 2),
                "rules": rules,
            })

        scored_candidates.sort(key=lambda item: item["score"], reverse=True)

        if mode == "strict":
            selected = scored_candidates[0]
        else:
            selected = random.choice(scored_candidates[:3])

        melody_before_choice = melody[:]
        melody.append(selected["note"])
        durations.append(choose_duration(step_index, length))

        if return_trace:
            trace.append({
                "step": step_index + 1,
                "melody_so_far": melody_before_choice,
                "possible_next_notes": scored_candidates[:5],
                "selected_note": selected["note"],
                "selected_score": selected["score"],
                "explanation": make_explanation(selected),
                "style_key": key,
                "style_name": settings["display_name"],
                "pitch_classes": pitch_classes,
            })

    return {
        "pitches": melody,
        "durations": durations,
        "trace": trace,
        "style_key": key,
        "style_name": settings["display_name"],
        "pitch_classes": pitch_classes,
        "home_note": home_note,
        "candidate_notes": candidate_notes,
    }


# ------------------------------------------------------------
# App-friendly class wrapper
# ------------------------------------------------------------

class RuleBasedMelodyGenerator:
    """
    Wrapper class so app.py can use the rule-based model like the other models.
    """

    def __init__(self, training_melody_key="minuet", mode="strict", seed=None):
        self.training_melody_key = training_melody_key
        self.mode = mode
        self.seed = seed
        self.last_trace = []
        self.last_result = None

        self.style_key = get_style_for_training_melody(training_melody_key)
        self.settings = STYLE_SETTINGS.get(self.style_key, STYLE_SETTINGS["G"])
        self.pitch_classes = self.settings["pitch_classes"]
        self.home_note = self.settings["home_note"]
        self.candidate_notes = build_candidate_notes(
            self.pitch_classes,
            self.settings["min_midi"],
            self.settings["max_midi"],
        )

    def generate_melody(self, length=16):
        result = generate_rule_based_melody(
            training_melody_key=self.training_melody_key,
            length=length,
            mode=self.mode,
            return_trace=True,
            seed=self.seed,
        )

        self.last_result = result
        self.last_trace = result["trace"]

        return list(zip(result["pitches"], result["durations"]))

    def get_trace_for_step(self, step):
        """
        Return trace information for a generated melody step.
        step is zero-based. Step 0 is the starting home note, so there is no
        candidate-scoring trace yet.
        """
        if step <= 0:
            return None

        trace_index = step - 1
        if 0 <= trace_index < len(self.last_trace):
            return self.last_trace[trace_index]

        return None


if __name__ == "__main__":
    for melody_key in ["minuet", "grace", "sakura"]:
        result = generate_rule_based_melody(
            training_melody_key=melody_key,
            length=16,
            mode="creative",
            seed=1,
        )
        print("\n", melody_key, "->", result["style_name"], result["pitch_classes"])
        print(result["pitches"])
