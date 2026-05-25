# Training melodies module

# Training melody: Minuet in G (Bach)
# Pitch encoding: {letter}{optional #}{octave} (e.g., G4, F#5)
# Durations in beats

MINUET_PITCHES = [
    "D5", "G4", "A4", "B4", "C5", "D5", "G4", "G4", "E5", "C5", "D5", "E5", "F#5", "G5", "G4", "G4",
    "C5", "D5", "C5", "B4", "A4", "B4", "C5", "B4", "A4", "G4", "F#4", "G4", "A4", "B4", "G4", "A4",
    "D5", "G4", "A4", "B4", "C5", "D5", "G4", "G4", "E5", "C5", "D5", "E5", "F#5", "G5", "G4", "G4",
    "C5", "D5", "C5", "B4", "A4", "B4", "C5", "B4", "A4", "G4", "A4", "B4", "A4", "G4", "F#4", "G4",
    "B4", "G4", "A4", "B4", "G4", "A4",
    "D5", "E5", "F#5", "D5",
    "G5", "E5", "F#5", "G5", "D5",
    "C#5", "B4", "C#5", "A4",
    "A4", "B4", "C#5", "D5", "E5", "F#5",
    "G5", "F#5", "E5", "F#5",
    "A4", "C#5", "D5",
    "D5", "G4", "F#4", "G4",
    "E5", "G4", "F#4", "G4",
    "D5", "C5", "B4",
    "A4", "G4", "F#4", "G4", "A4",
    "D4", "E4", "F#4", "G4", "A4", "B4",
    "C5", "B4", "A4",
    "B4", "D5", "G4", "F#4", "G4"
]

MINUET_RHYTHM = [
    1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1,
    1, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 3,
    1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1,
    1, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 3,
    1, 0.5, 0.5, 0.5, 0.5, 1,
    0.5, 0.5, 0.5, 0.5,
    1, 0.5, 0.5, 0.5, 0.5,
    1, 0.5, 0.5, 1,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    1, 1, 1, 1,
    1, 1, 3,
    1, 0.5, 0.5, 1,
    1, 0.5, 0.5, 1,
    1, 1, 1,
    0.5, 0.5, 0.5, 0.5, 1,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    1, 1, 1,
    0.5, 0.5, 1, 1,
    3
]

assert len(MINUET_PITCHES) == len(MINUET_RHYTHM), "Training pitch and rhythm streams must be aligned."

# Training melody: Amazing Grace
AMAZING_GRACE_PITCHES = [
    "C4", "F4", "A4", "F4", "A4", "G4", "F4", "D4", "C4",
    "C4", "F4", "A4", "F4", "A4", "G4", "C5",
    "A4", "C5", "REST", "A4", "C5", "A4", "F4",
    "C4", "D4", "REST", "F4", "F4", "D4", "C4",
    "C4", "F4", "A4", "F4", "A4", "G4", "F4"
]

AMAZING_GRACE_RHYTHM = [
    1, 2, 0.5, 0.5, 2, 1, 2, 1, 2,
    1, 2, 0.5, 0.5, 2, 1, 4,
    1, 1, 0.5, 0.5, 0.5, 0.5, 2,
    1, 1, 0.5, 0.5, 0.5, 0.5, 2,
    1, 2, 0.5, 0.5, 2, 1, 3
]

assert len(AMAZING_GRACE_PITCHES) == len(AMAZING_GRACE_RHYTHM), "Training pitch and rhythm streams must be aligned."

# Training melody: Sakura Sakura
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
    "minuet": {
        "name": "Minuet (Bach)",
        "pitches": MINUET_PITCHES,
        "rhythms": MINUET_RHYTHM
    },
    "grace": {
        "name": "Amazing Grace",
        "pitches": AMAZING_GRACE_PITCHES,
        "rhythms": AMAZING_GRACE_RHYTHM
    },
    "sakura": {
        "name": "Sakura Sakura",
        "pitches": SAKURA_PITCHES,
        "rhythms": SAKURA_RHYTHM
    }
}