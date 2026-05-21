import math
from collections import Counter, defaultdict

# Pitch-class mapping for common note spellings
PCLASS = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}

MOTIF_LENGTHS = (2, 3, 4)


def normalize_seq(melody):
    """
    Normalize a melody sequence.
    Expects a list of [pitch, duration] tokens.
    - Converts durations to float.
    - Supports rests as 'REST' or 'rest'.
    """
    normalized = []
    for token in melody:
        if not isinstance(token, (list, tuple)) or len(token) != 2:
            raise ValueError(f"Each token must be [pitch, dur]; got: {token!r}")
        note, dur = token
        if note is None:
            raise ValueError("Note cannot be None")
        note_s = str(note).strip()
        try:
            dur_f = float(dur)
        except Exception as e:
            raise ValueError(f"Duration must be numeric for token {token!r}: {e}")
        normalized.append([note_s, dur_f])
    return normalized


def pitch_stream(seq):
    """Return the sequence of pitch names (ignoring durations)."""
    return [token[0] for token in seq]


def durations_numeric(seq):
    """Return the sequence of numeric durations."""
    return [token[1] for token in seq]


def is_rest(note):
    """Check if a note is a rest."""
    return str(note).upper().startswith("REST")


def note_to_number(note):
    """
    Convert a pitch string to MIDI number (C4 = 60).
    Supports sharps (#) and flats (b).
    Octave must be specified at the end.
    Rests return None.
    """
    s = str(note).strip()
    
    if is_rest(s):
        return None
    
    if not s:
        raise ValueError("Empty note string")
    
    # Find where the octave number starts
    i = len(s)
    while i > 0 and s[i - 1].isdigit():
        i -= 1
    
    name = s[:i]
    if i < len(s):
        try:
            octv = int(s[i:])
        except ValueError:
            raise ValueError(f"Invalid octave in note string: {note!r}")
    else:
        raise ValueError(f"Octave required in note string: {note!r}")
    
    if name not in PCLASS:
        raise ValueError(f"Unknown or unsupported pitch name: {name!r}")
    
    semitone_offset = PCLASS[name]
    return 60 + semitone_offset + 12 * (octv - 4)


def midi_stream(seq):
    """Convert pitch sequence to MIDI numbers, skipping rests."""
    midi_nums = []
    for p in pitch_stream(seq):
        if not is_rest(p):
            midi_nums.append(note_to_number(p))
    return midi_nums


def intervals_from_midi(midi_nums):
    """Compute intervals (in semitones) between consecutive MIDI notes."""
    if len(midi_nums) < 2:
        return []
    return [midi_nums[i + 1] - midi_nums[i] for i in range(len(midi_nums) - 1)]


def normalize_base_name(pitch):
    """Return the pitch class name without octave."""
    s = str(pitch).strip()
    
    if is_rest(s):
        return "REST"
    
    i = len(s)
    while i > 0 and s[i - 1].isdigit():
        i -= 1
    return s[:i]


def infer_tonic_from_last8(pitches):
    """
    Infer tonic heuristically from pitch-class counts in the final 8 notes.
    Limited to G major and D major based on pitch patterns.
    """
    if not pitches:
        return None
    
    last8 = pitches[-8:]
    base_counts = Counter(normalize_base_name(p) for p in last8 if not is_rest(p))
    
    # D major heuristic: C#, D, A or F#
    cond_csharp = base_counts.get("C#", 0) >= 2
    cond_d = base_counts.get("D", 0) >= 2
    cond_a_or_fsharp = (base_counts.get("A", 0) >= 1) or (base_counts.get("F#", 0) >= 1)
    
    return "D" if (cond_csharp and cond_d and cond_a_or_fsharp) else "G"


def ending_on_tonic(seq):
    """Check if melody ends on the inferred tonic."""
    pitches = pitch_stream(seq)
    non_rest_pitches = [p for p in pitches if not is_rest(p)]
    
    if not non_rest_pitches:
        return 0.0, {"tonic": None, "final_note": None}
    
    tonic = infer_tonic_from_last8(pitches)
    final_note = normalize_base_name(non_rest_pitches[-1])
    score = 1.0 if final_note == tonic else 0.0
    
    return score, {"tonic": tonic, "final_note": non_rest_pitches[-1]}


def _nontrivial_pattern(values):
    """A pattern is non-trivial if it contains at least two distinct values."""
    return len(set(values)) >= 2


def _greedy_non_overlapping_starts(starts, length):
    """Select a maximal non-overlapping subset of sorted start indices."""
    chosen = []
    next_allowed = -1
    for start in sorted(starts):
        if start >= next_allowed:
            chosen.append(start)
            next_allowed = start + length
    return chosen


def find_repeated_non_overlapping_patterns(stream, lengths=MOTIF_LENGTHS):
    """Identify repeated non-trivial subsequences of requested lengths."""
    n = len(stream)
    length_counts = {L: 0 for L in lengths}
    covered_positions = set()
    details = {str(L): [] for L in lengths}
    
    for L in lengths:
        if n < L:
            continue
        
        occurrences = defaultdict(list)
        for i in range(n - L + 1):
            subseq = tuple(stream[i : i + L])
            if not _nontrivial_pattern(subseq):
                continue
            occurrences[subseq].append(i)
        
        for subseq, starts in occurrences.items():
            chosen_starts = _greedy_non_overlapping_starts(starts, L)
            if len(chosen_starts) < 2:
                continue
            
            length_counts[L] += 1
            for start in chosen_starts:
                covered_positions.update(range(start, start + L))
    
    return length_counts, covered_positions, details


def pattern_strength(length_counts, denominator):
    """Pattern-strength term: min(1, sum(L * count) / (4 * denominator))."""
    if denominator <= 0:
        return 0.0
    numerator = sum(L * length_counts.get(L, 0) for L in MOTIF_LENGTHS)
    return min(1.0, numerator / (4.0 * denominator))


def interval_smoothness(intervals):
    """Continuous interval smoothness: 1 - (large jumps / total intervals)."""
    n_int = len(intervals)
    if n_int == 0:
        return 0.0, {"n_large": 0, "n_int": 0}
    
    n_large = sum(1 for value in intervals if abs(value) > 7)
    score = 1.0 - (n_large / n_int)
    return score, {"n_large": n_large, "n_int": n_int}


def stepwise_motion(intervals):
    """Continuous stepwise-motion: (stepwise intervals / total intervals)."""
    n_int = len(intervals)
    if n_int == 0:
        return 0.0, {"n_step": 0, "n_int": 0}
    
    n_step = sum(1 for value in intervals if abs(value) <= 2)
    score = n_step / n_int
    return score, {"n_step": n_step, "n_int": n_int}


def pitch_motif_score(seq):
    """Pitch-motif score: 0.6 * coverage + 0.4 * pattern_strength."""
    pitches = [p for p in pitch_stream(seq) if not is_rest(p)]
    N = len(pitches)
    if N == 0:
        return 0.0, {}
    
    length_counts, covered_positions, _ = find_repeated_non_overlapping_patterns(pitches)
    coverage = len(covered_positions) / N
    ps = pattern_strength(length_counts, N)
    score = 0.6 * coverage + 0.4 * ps
    
    return score, {
        "coverage": coverage,
        "pattern_strength": ps,
        "length_counts": length_counts,
    }


def interval_motif_score(intervals):
    """Interval-motif score: 0.6 * coverage + 0.4 * pattern_strength."""
    denom = len(intervals)
    if denom == 0:
        return 0.0, {}
    
    length_counts, covered_positions, _ = find_repeated_non_overlapping_patterns(intervals)
    coverage = len(covered_positions) / denom
    ps = pattern_strength(length_counts, denom)
    score = 0.6 * coverage + 0.4 * ps
    
    return score, {
        "coverage": coverage,
        "pattern_strength": ps,
        "length_counts": length_counts,
    }


def motif_repetition(seq, intervals):
    """Overall motif repetition: (pitch_score + interval_score) / 2."""
    pitch_score, pitch_details = pitch_motif_score(seq)
    interval_score, interval_details = interval_motif_score(intervals)
    score = 0.5 * (pitch_score + interval_score)
    
    return score, {
        "pitch_score": pitch_score,
        "interval_score": interval_score,
    }


def infer_duration_vocab_size(melodies_list):
    """Infer D_max from all distinct duration values in melodies."""
    all_durations = set()
    for melody in melodies_list:
        seq = normalize_seq(melody)
        all_durations.update(durations_numeric(seq))
    return max(1, len(all_durations))


def normalized_rhythmic_entropy(durations, d_max):
    """Normalized rhythmic entropy: -sum(p_i * log(p_i)) / log(D_max)."""
    if not durations or d_max <= 1:
        return 0.0
    
    total = len(durations)
    counts = Counter(durations)
    entropy = 0.0
    for count in counts.values():
        p_i = count / total
        entropy -= p_i * math.log(p_i)
    
    return entropy / math.log(d_max)


def rhythmic_pattern_score(durations):
    """Rhythmic pattern score: 0.6 * coverage + 0.4 * pattern_strength."""
    N = len(durations)
    if N == 0:
        return 0.0, {}
    
    length_counts, covered_positions, _ = find_repeated_non_overlapping_patterns(durations)
    coverage = len(covered_positions) / N
    ps = pattern_strength(length_counts, N)
    score = 0.6 * coverage + 0.4 * ps
    
    return score, {
        "coverage": coverage,
        "pattern_strength": ps,
    }


def rhythmic_variety(durations, d_max):
    """Rhythmic variety: 0.5 * H_norm + 0.5 * RPS."""
    h_norm = normalized_rhythmic_entropy(durations, d_max)
    rps, rps_details = rhythmic_pattern_score(durations)
    score = 0.5 * h_norm + 0.5 * rps
    
    return score, {
        "H_norm": h_norm,
        "RPS": rps,
    }


def evaluate_sequence(melody, d_max):
    """
    Compute all evaluation metrics for a melody.
    Returns a dictionary with all scores and metadata.
    """
    seq = normalize_seq(melody)
    midi_seq = midi_stream(seq)
    intervals = intervals_from_midi(midi_seq)
    durations = durations_numeric(seq)
    
    end_score, end_meta = ending_on_tonic(seq)
    smooth_score, smooth_meta = interval_smoothness(intervals)
    step_score, step_meta = stepwise_motion(intervals)
    motif_score, motif_meta = motif_repetition(seq, intervals)
    rhythm_score, rhythm_meta = rhythmic_variety(durations, d_max)
    
    final_score = (end_score + smooth_score + step_score + motif_score + rhythm_score) / 5.0
    
    return {
        "ending_on_tonic": end_score,
        "interval_smoothness": smooth_score,
        "stepwise_ratio": step_score,
        "motif_repetition": motif_score,
        "rhythmic_variety": rhythm_score,
        "final_score": final_score,
        "meta": {
            "ending_on_tonic": end_meta,
            "interval_smoothness": smooth_meta,
            "stepwise_ratio": step_meta,
            "motif_repetition": motif_meta,
            "rhythmic_variety": rhythm_meta,
        },
    }


def categorize_score(score, category_type="standard"):
    """
    Categorize a numerical score into qualitative categories.
    
    Args:
        score: Float between 0 and 1
        category_type: "standard" for High/Med/Low, "rhythm" for Varied/Balanced/Repetitive
    
    Returns:
        Tuple of (category_name, bar_filled)
    """
    if category_type == "rhythm":
        if score >= 0.7:
            return "Varied", 7
        elif score >= 0.4:
            return "Balanced", 5
        else:
            return "Repetitive", 3
    else:  # standard
        if score >= 0.7:
            return "High", 7
        elif score >= 0.4:
            return "Med", 5
        else:
            return "Low", 3


def ending_category(score):
    """
    Categorize ending on tonic (binary).
    
    Returns:
        Tuple of (category_name, bar_filled)
    """
    if score >= 0.5:
        return "Finished on Tonic", 10
    else:
        return "Unresolved Ending", 0
