"""
Preprocess MelodyHub for MelodyLab.

This script:
1. Loads the MelodyHub dataset from Hugging Face.
2. Keeps only generation-task rows.
3. Parses ABC notation from row["output"].
4. Converts melodies into MelodyLab note events: [("C4", 1.0), ...].
5. Creates two JSON style packs:
   - Beginner Pack: strict filters
   - Public Domain Pack: looser filters

Run:
    python scripts/preprocess_melodyhub.py
"""

# rests are skipped

from datasets import load_dataset
from music21 import converter, note, chord
from tqdm import tqdm
import json
import os
from collections import Counter


# -----------------------------
# Settings
# -----------------------------

BEGINNER_TARGET = 1000
PUBLIC_DOMAIN_TARGET = 3000

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BEGINNER_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "beginner_pack.json")
PUBLIC_DOMAIN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "public_domain_pack.json")


# Beginner Pack: strict, clean melodies
BEGINNER_FILTERS = {
    "min_notes": 16,
    "max_notes": 256,
    "max_rest_ratio": 0.25,
    "max_weird_rhythm_ratio": 0.15,
    "max_accidental_ratio": 0.30,
    "max_pitch_range": 36,
    "allow_chords": False,
}


# Public Domain Pack: looser, more varied melodies
PUBLIC_DOMAIN_FILTERS = {
    "min_notes": 16,
    "max_notes": 1024,
    "max_rest_ratio": 0.35,
    "allow_chords": False,
}


ALLOWED_RHYTHMS = {
    0.25,  # sixteenth note
    0.5,   # eighth note
    1.0,   # quarter note
    1.5,   # dotted quarter
    2.0,   # half note
    3.0,   # dotted half
    4.0,   # whole note
}


# -----------------------------
# Helpers
# -----------------------------

def normalize_duration(duration):
    """
    Convert music21 duration to a normal Python float.

    music21 sometimes gives fractions or special duration objects.
    """
    try:
        return float(duration.quarterLength)
    except Exception:
        return None


def is_weird_rhythm(duration_value):
    """
    A rhythm is considered weird if it is not close to one of the allowed values.
    Small floating-point errors are tolerated.
    """
    if duration_value is None:
        return True

    tolerance = 0.001

    for allowed in ALLOWED_RHYTHMS:
        if abs(duration_value - allowed) <= tolerance:
            return False

    return True


def pitch_has_accidental(pitch_obj):
    """
    Returns True if a pitch has a sharp, flat, natural alteration, etc.
    """
    return pitch_obj.accidental is not None


def safe_parse_abc(abc_text):
    """
    Try parsing ABC notation using music21.
    Returns the parsed score, or None if parsing fails.
    """
    try:
        return converter.parse(abc_text, format="abc")
    except Exception:
        return None


def extract_events_and_stats(score):
    """
    Extract MelodyLab note events and useful filtering stats from a parsed score.

    MelodyLab event format:
        ["C4", 1.0]

    We store lists instead of tuples because JSON does not preserve tuples.
    """

    events = []

    note_count = 0
    rest_count = 0
    chord_count = 0
    weird_rhythm_count = 0
    accidental_count = 0
    midi_values = []

    for element in score.recurse().notesAndRests:
        duration_value = normalize_duration(element.duration)

        if isinstance(element, note.Rest):
            rest_count += 1

            if is_weird_rhythm(duration_value):
                weird_rhythm_count += 1

            # For now, rests are counted for filtering but not included in MelodyLab tokens.
            continue

        if isinstance(element, chord.Chord):
            chord_count += 1

            if is_weird_rhythm(duration_value):
                weird_rhythm_count += 1

            # MelodyLab is melody-only for now, so chord note blocks are not converted.
            continue

        if isinstance(element, note.Note):
            note_count += 1

            if is_weird_rhythm(duration_value):
                weird_rhythm_count += 1

            if pitch_has_accidental(element.pitch):
                accidental_count += 1

            midi_values.append(element.pitch.midi)

            pitch_name = element.pitch.nameWithOctave
            rhythm = duration_value

            events.append([pitch_name, rhythm])

    total_sound_events = note_count + chord_count
    total_events_with_rests = note_count + chord_count + rest_count

    if note_count > 0:
        accidental_ratio = accidental_count / note_count
    else:
        accidental_ratio = 0

    if total_events_with_rests > 0:
        rest_ratio = rest_count / total_events_with_rests
        weird_rhythm_ratio = weird_rhythm_count / total_events_with_rests
    else:
        rest_ratio = 0
        weird_rhythm_ratio = 0

    if midi_values:
        pitch_range = max(midi_values) - min(midi_values)
    else:
        pitch_range = 0

    stats = {
        "note_count": note_count,
        "rest_count": rest_count,
        "chord_count": chord_count,
        "total_sound_events": total_sound_events,
        "total_events_with_rests": total_events_with_rests,
        "rest_ratio": rest_ratio,
        "weird_rhythm_ratio": weird_rhythm_ratio,
        "accidental_ratio": accidental_ratio,
        "pitch_range": pitch_range,
    }

    return events, stats


def passes_beginner_filters(stats):
    """
    Strict filters for clean, beginner-friendly melodies.
    """

    if stats["note_count"] < BEGINNER_FILTERS["min_notes"]:
        return False, "too_short"

    if stats["note_count"] > BEGINNER_FILTERS["max_notes"]:
        return False, "too_long"

    if not BEGINNER_FILTERS["allow_chords"] and stats["chord_count"] > 0:
        return False, "has_chords"

    if stats["rest_ratio"] > BEGINNER_FILTERS["max_rest_ratio"]:
        return False, "too_many_rests"

    if stats["weird_rhythm_ratio"] > BEGINNER_FILTERS["max_weird_rhythm_ratio"]:
        return False, "too_many_weird_rhythms"

    if stats["accidental_ratio"] > BEGINNER_FILTERS["max_accidental_ratio"]:
        return False, "too_many_accidentals"

    if stats["pitch_range"] > BEGINNER_FILTERS["max_pitch_range"]:
        return False, "pitch_range_too_large"

    return True, "passed"


def passes_public_domain_filters(stats):
    """
    Looser filters for the public domain melody pack.

    This keeps:
    - failed parsing removal
    - chord/polyphony removal
    - too short removal
    - too long removal
    - too many rests removal

    This skips:
    - weird rhythm filtering
    - accidental/chromatic filtering
    - pitch range filtering
    """

    if stats["note_count"] < PUBLIC_DOMAIN_FILTERS["min_notes"]:
        return False, "too_short"

    if stats["note_count"] > PUBLIC_DOMAIN_FILTERS["max_notes"]:
        return False, "too_long"

    if not PUBLIC_DOMAIN_FILTERS["allow_chords"] and stats["chord_count"] > 0:
        return False, "has_chords"

    if stats["rest_ratio"] > PUBLIC_DOMAIN_FILTERS["max_rest_ratio"]:
        return False, "too_many_rests"

    return True, "passed"


def make_melody_record(index, events, stats, source_dataset):
    """
    Create one clean melody record for the JSON file.
    """

    return {
        "name": f"melodyhub_{index}",
        "source": "MelodyHub",
        "source_dataset": source_dataset,
        "events": events,
        "stats": {
            "note_count": stats["note_count"],
            "rest_count": stats["rest_count"],
            "chord_count": stats["chord_count"],
            "rest_ratio": round(stats["rest_ratio"], 4),
            "weird_rhythm_ratio": round(stats["weird_rhythm_ratio"], 4),
            "accidental_ratio": round(stats["accidental_ratio"], 4),
            "pitch_range": stats["pitch_range"],
        },
    }


def save_pack(path, name, description, melodies):
    """
    Save a style pack as JSON.
    """

    pack = {
        "name": name,
        "description": description,
        "format": "melodylab_event_pack_v1",
        "event_format": ["pitch", "rhythm"],
        "melody_count": len(melodies),
        "melodies": melodies,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2, ensure_ascii=False)


# -----------------------------
# Main preprocessing
# -----------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading MelodyHub...")
    ds = load_dataset("sander-wood/melodyhub")

    beginner_melodies = []
    public_domain_melodies = []

    beginner_reject_reasons = Counter()
    public_domain_reject_reasons = Counter()

    parse_failures = 0
    generation_rows_seen = 0

    print("Processing generation rows...")

    for row in tqdm(ds["train"]):
        if row["task"] != "generation":
            continue

        generation_rows_seen += 1

        abc_text = row["output"]
        source_dataset = row.get("dataset", "unknown")

        score = safe_parse_abc(abc_text)

        if score is None:
            parse_failures += 1
            continue

        events, stats = extract_events_and_stats(score)

        # If something parsed but produced no usable note events, skip it.
        if not events:
            beginner_reject_reasons["no_events"] += 1
            public_domain_reject_reasons["no_events"] += 1
            continue

        beginner_ok, beginner_reason = passes_beginner_filters(stats)
        public_ok, public_reason = passes_public_domain_filters(stats)

        if beginner_ok and len(beginner_melodies) < BEGINNER_TARGET:
            melody = make_melody_record(
                index=len(beginner_melodies),
                events=events,
                stats=stats,
                source_dataset=source_dataset,
            )
            beginner_melodies.append(melody)
        else:
            if not beginner_ok:
                beginner_reject_reasons[beginner_reason] += 1

        if public_ok and len(public_domain_melodies) < PUBLIC_DOMAIN_TARGET:
            melody = make_melody_record(
                index=len(public_domain_melodies),
                events=events,
                stats=stats,
                source_dataset=source_dataset,
            )
            public_domain_melodies.append(melody)
        else:
            if not public_ok:
                public_domain_reject_reasons[public_reason] += 1

        if (
            len(beginner_melodies) >= BEGINNER_TARGET
            and len(public_domain_melodies) >= PUBLIC_DOMAIN_TARGET
        ):
            break

    save_pack(
        path=BEGINNER_OUTPUT_PATH,
        name="Beginner Pack",
        description=(
            "A strict, clean MelodyHub subset with simple lengths, low rest usage, "
            "simple rhythms, limited accidentals, and limited pitch range."
        ),
        melodies=beginner_melodies,
    )

    save_pack(
        path=PUBLIC_DOMAIN_OUTPUT_PATH,
        name="Public Domain Melodies Pack",
        description=(
            "A looser MelodyHub subset of public-domain-style melodies. "
            "Keeps melody-only filtering but allows wider pitch ranges, chromaticism, "
            "and more complex rhythms."
        ),
        melodies=public_domain_melodies,
    )

    print("\nDone.")
    print(f"Generation rows seen: {generation_rows_seen}")
    print(f"Parse failures: {parse_failures}")

    print(f"\nBeginner Pack saved: {BEGINNER_OUTPUT_PATH}")
    print(f"Beginner melodies: {len(beginner_melodies)} / {BEGINNER_TARGET}")
    print("Beginner reject reasons:")
    for reason, count in beginner_reject_reasons.most_common():
        print(f"  {reason}: {count}")

    print(f"\nPublic Domain Pack saved: {PUBLIC_DOMAIN_OUTPUT_PATH}")
    print(f"Public domain melodies: {len(public_domain_melodies)} / {PUBLIC_DOMAIN_TARGET}")
    print("Public domain reject reasons:")
    for reason, count in public_domain_reject_reasons.most_common():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
