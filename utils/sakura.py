# Training melody: Sakura Sakura
# Pitch encoding: {letter}{optional #}{octave}
# Durations in beats

SAKURA_PITCHES = [
    "A4", "A4", "B4",
    "A4", "A4", "B4",
    "A4", "B4", "C5", "B4",
    "A4", "B4", "A4", "F4",
    "E4", "C4", "E4", "F4",
    "E4", "E4", "C4", "B3",

    "A4", "B4", "C5", "B4",
    "A4", "B4", "A4", "F4",
    "E4", "C4", "E4", "F4",
    "E4", "E4", "C4", "B3",

    "A4", "A4", "B4",
    "A4", "A4", "B4",

    "E4", "F4", "B4", "A4", "F4",
    "E4"
]

SAKURA_RHYTHM = [
    1, 1, 2,
    1, 1, 2,

    1, 1, 1, 1,
    1, 0.5, 0.5, 2,
    1, 1, 1, 1,
    1, 0.5, 0.5, 2,

    1, 1, 1, 1,
    1, 0.5, 0.5, 2,
    1, 1, 1, 1,
    1, 0.5, 0.5, 2,

    1, 1, 2,
    1, 1, 2,

    1, 1, 0.5, 0.5, 1,
    4
]

assert len(SAKURA_PITCHES) == len(SAKURA_RHYTHM), "Training pitch and rhythm streams must be aligned."

TRAINING_MELODIES = {
    "sakura": {
        "name": "Sakura Sakura",
        "pitches": SAKURA_PITCHES,
        "rhythms": SAKURA_RHYTHM
    }
}
