# rule_based_model.py

import random


# ------------------------------------------------------------
# Note-to-MIDI mapping
# ------------------------------------------------------------
# MIDI numbers let us measure the distance between notes.
# Example:
# G4 = 67, A4 = 69
# Difference = 2 semitones, so G4 -> A4 is stepwise motion.

NOTE_TO_MIDI = {
    "C4": 60,
    "D4": 62,
    "E4": 64,
    "F4": 65,
    "F#4": 66,
    "G4": 67,
    "A4": 69,
    "Bb4": 70,
    "B4": 71,

    "C5": 72,
    "D5": 74,
    "E5": 76,
    "F5": 77,
    "F#5": 78,
    "G5": 79,
    "A5": 81,
    "B5": 83,
}


# ------------------------------------------------------------
# Scale settings
# ------------------------------------------------------------
# The rule-based model does not "train" on the melody.
# But it can use the selected melody's musical world:
# Minuet -> G major
# Amazing Grace -> F major
# Sakura -> A-based Sakura pentatonic

SCALES = {
    "G": ["G4", "A4", "B4", "C5", "D5", "E5", "F#5", "G5"],

    "F": ["F4", "G4", "A4", "Bb4", "C5", "D5", "E5", "F5"],

    "C": ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],

    "A_SAKURA": ["A4", "B4", "C5", "E5", "F5", "A5"],
}


# ------------------------------------------------------------
# Rhythm options
# ------------------------------------------------------------
# 1.0 appears twice, so one-beat notes are slightly more common.

DEFAULT_DURATIONS = [0.5, 1.0, 1.0, 1.5, 2.0]


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def pitch_distance(note1, note2):
    """
    Returns the absolute pitch distance between two notes.

    Example:
    G4 -> A4 = 2
    A4 -> G4 = 2

    Direction is ignored.
    """

    return abs(NOTE_TO_MIDI[note2] - NOTE_TO_MIDI[note1])


def signed_pitch_distance(note1, note2):
    """
    Returns the signed pitch distance between two notes.

    Example:
    G4 -> A4 = +2
    A4 -> G4 = -2

    Direction is preserved.
    """

    return NOTE_TO_MIDI[note2] - NOTE_TO_MIDI[note1]


# ------------------------------------------------------------
# Candidate scoring function
# ------------------------------------------------------------

def score_candidate(candidate, melody_so_far, scale_notes, home_note, step_index, length):
    """
    Scores one possible next note according to music-theory rules.

    Higher score = better candidate.

    Rules used:
    1. Stay in scale
    2. Prefer stepwise motion
    3. Avoid too much repetition
    4. Resolve large leaps
    5. Stay in comfortable range
    6. Move toward / end on home note
    """

    score = 0
    rules = []

    last_note = melody_so_far[-1]

    # --------------------------------------------------------
    # Rule 1: Stay in scale
    # --------------------------------------------------------

    if candidate in scale_notes:
        score += 3
        rules.append("stays in the scale")
    else:
        score -= 5
        rules.append("outside the scale")

    # --------------------------------------------------------
    # Rule 2: Prefer stepwise motion
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Rule 3: Avoid too much repetition
    # --------------------------------------------------------

    if candidate == last_note:
        score -= 2
        rules.append("repeats the previous note")

    if len(melody_so_far) >= 2:
        if candidate == melody_so_far[-1] == melody_so_far[-2]:
            score -= 5
            rules.append("avoids three repeated notes in a row")

    # --------------------------------------------------------
    # Rule 4: Resolve large leaps
    # --------------------------------------------------------
    # If the previous movement was a big leap, the next note
    # should ideally move stepwise in the opposite direction.

    if len(melody_so_far) >= 2:
        previous_interval = signed_pitch_distance(
            melody_so_far[-2],
            melody_so_far[-1]
        )

        next_interval = signed_pitch_distance(
            melody_so_far[-1],
            candidate
        )

        if abs(previous_interval) > 5:
            moves_opposite_direction = previous_interval * next_interval < 0
            moves_stepwise = abs(next_interval) <= 2

            if moves_opposite_direction and moves_stepwise:
                score += 4
                rules.append("resolves the previous leap by moving stepwise in the opposite direction")
            else:
                score -= 3
                rules.append("does not resolve the previous leap clearly")

    # --------------------------------------------------------
    # Rule 5: Keep melody in a comfortable range
    # --------------------------------------------------------

    midi_value = NOTE_TO_MIDI[candidate]

    if 60 <= midi_value <= 79:
        score += 1
        rules.append("keeps the melody in a comfortable range")
    else:
        score -= 2
        rules.append("moves outside the preferred range")

    # --------------------------------------------------------
    # Rule 6: Ending / home note rule
    # --------------------------------------------------------

    is_final_note = step_index == length - 1

    if is_final_note:
        if candidate == home_note:
            score += 10
            rules.append("ends on the home note")
        else:
            score -= 4
            rules.append("does not end on the home note")

    else:
        # Near the end, prefer notes close to the home note.
        remaining_notes = length - step_index - 1

        if remaining_notes <= 3:
            distance_to_home = pitch_distance(candidate, home_note)

            if distance_to_home <= 4:
                score += 2
                rules.append("moves closer to the home note near the ending")

    return score, rules


# ------------------------------------------------------------
# Rhythm function
# ------------------------------------------------------------

def choose_duration(step_index, length):
    """
    Chooses a duration for the current note.

    The final note is longer so the melody feels finished.
    """

    is_final_note = step_index == length - 1

    if is_final_note:
        return 2.0

    return random.choice(DEFAULT_DURATIONS)


# ------------------------------------------------------------
# Explanation generator
# ------------------------------------------------------------

def make_explanation(selected):
    """
    Converts the selected note's rule list into a readable sentence.
    """

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
        return (
            f"{selected['note']} was selected because it "
            + ", ".join(useful_rules[:3])
            + "."
        )

    return f"{selected['note']} was selected because it had the best overall rule score."


# ------------------------------------------------------------
# Main rule-based melody generator
# ------------------------------------------------------------

def generate_rule_based_melody(
    key="G",
    length=16,
    home_note=None,
    scale_notes=None,
    mode="strict",
    return_trace=True,
    seed=None
):
    """
    Generates a melody using explicit music-theory rules.

    Parameters
    ----------
    key : str
        Musical key/scale name.
        Examples: "G", "F", "C", "A_SAKURA"

    length : int
        Number of notes to generate.

    home_note : str or None
        The tonic/home note.
        If None, the first note of the scale is used.

    scale_notes : list or None
        Custom scale notes.
        If None, notes are taken from SCALES using the selected key.

    mode : str
        "strict"   -> always choose the highest-scoring note.
        "creative" -> randomly choose from the top 3 candidates.

    return_trace : bool
        If True, returns note-by-note reasoning for Learning Mode.

    seed : int or None
        Optional random seed for reproducible output.

    Returns
    -------
    dict with:
        "pitches"   : generated pitch list
        "durations" : generated duration list
        "trace"     : note-by-note explanation data
    """

    if seed is not None:
        random.seed(seed)

    if scale_notes is None:
        scale_notes = SCALES.get(key, SCALES["G"])

    if home_note is None:
        home_note = scale_notes[0]

    candidate_notes = scale_notes[:]

    melody = [home_note]
    durations = [1.0]
    trace = []

    for step_index in range(1, length):
        scored_candidates = []

        # Score every possible next note.
        for candidate in candidate_notes:
            score, rules = score_candidate(
                candidate=candidate,
                melody_so_far=melody,
                scale_notes=scale_notes,
                home_note=home_note,
                step_index=step_index,
                length=length
            )

            scored_candidates.append({
                "note": candidate,
                "score": round(score, 2),
                "rules": rules
            })

        # Sort from best candidate to worst candidate.
        scored_candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        # Choose next note.
        if mode == "strict":
            selected = scored_candidates[0]
        elif mode == "creative":
            top_candidates = scored_candidates[:3]
            selected = random.choice(top_candidates)
        else:
            raise ValueError("mode must be either 'strict' or 'creative'")

        # Store melody before adding selected note.
        melody_before_choice = melody[:]

        # Add selected pitch and duration.
        melody.append(selected["note"])
        durations.append(choose_duration(step_index, length))

        # Store trace for Learning Mode.
        if return_trace:
            trace.append({
                "step": step_index + 1,
                "melody_so_far": melody_before_choice,
                "possible_next_notes": scored_candidates[:5],
                "selected_note": selected["note"],
                "selected_score": selected["score"],
                "explanation": make_explanation(selected)
            })

    return {
        "pitches": melody,
        "durations": durations,
        "trace": trace
    }


# ------------------------------------------------------------
# Quick test
# ------------------------------------------------------------

if __name__ == "__main__":
    result = generate_rule_based_melody(
        key="G",
        length=16,
        mode="strict",
        return_trace=True
    )

    print("Generated pitches:")
    print(result["pitches"])

    print("\nGenerated durations:")
    print(result["durations"])

    print("\nFirst trace step:")
    print(result["trace"][0])