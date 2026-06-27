import streamlit as st
import numpy as np
import io
import wave
import base64
import json
from pathlib import Path
from utils.melodies import TRAINING_MELODIES
from utils.scorecard import evaluate_sequence, infer_duration_vocab_size
from models.random_model import RandomMelodyGenerator
from models.weighted_random import WeightedRandomMelodyGenerator
from models.markov1 import FirstOrderMarkovGenerator
from models.markov2 import SecondOrderMarkovGenerator
from models.variable_markov import VariableMarkovGenerator
from models.rule_based_basic import RuleBasedMelodyGenerator

APP_DIR = Path(__file__).resolve().parent
LOGO_FILE = APP_DIR / "melodylab_logo.png"
FAVICON_FILE = APP_DIR / "melodylab_favicon.png"
DATA_DIR = APP_DIR / "data" / "processed"
if not DATA_DIR.exists():
    DATA_DIR = Path("C:/Users/Poonam-hp/Documents/GitHub/MelodyLab/data/processed")


if not LOGO_FILE.exists():
    LOGO_FILE = Path("C:/Users/Poonam-hp/Documents/GitHub/MelodyLab/melodylab_logo.png")
if not FAVICON_FILE.exists():
    FAVICON_FILE = Path("C:/Users/Poonam-hp/Documents/GitHub/MelodyLab/melodylab_favicon.png")

st.set_page_config(
    page_title="MelodyLab",
    page_icon=FAVICON_FILE if FAVICON_FILE.exists() else LOGO_FILE if LOGO_FILE.exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "selected_melody" not in st.session_state:
    st.session_state.selected_melody = None
if "generated_melody" not in st.session_state:
    st.session_state.generated_melody = None
if "model_instance" not in st.session_state:
    st.session_state.model_instance = None
if "scorecard_results" not in st.session_state:
    st.session_state.scorecard_results = None
if "learning_step" not in st.session_state:
    st.session_state.learning_step = 0
if "rule_mode" not in st.session_state:
    st.session_state.rule_mode = None
if "play_learning_note" not in st.session_state:
    st.session_state.play_learning_note = False
if "melody_length" not in st.session_state:
    st.session_state.melody_length = 16
if "melody_history" not in st.session_state:
    st.session_state.melody_history = []
if "saved_history_index" not in st.session_state:
    st.session_state.saved_history_index = None

MELODY_ICONS = {
    "minuet": "🎻",
    "amazing_grace": "🎺",
    "amazing grace": "🎺",
    "sakura": "🌸",
    "sakura_sakura": "🌸",
    "sakura sakura": "🌸",
    "raga": "🪷",
    "indian": "🪷",
}

MIN_MELODY_LENGTH = 8
MAX_MELODY_LENGTH = 32


def get_logo_markup():
    """Return the uploaded MelodyLab logo image, with a text fallback."""
    if LOGO_FILE.exists():
        logo_data = base64.b64encode(LOGO_FILE.read_bytes()).decode("utf-8")
        return f'<img class="ml-logo-img" src="data:image/png;base64,{logo_data}" alt="MelodyLab logo" />'
    return '<div class="ml-logo">ML</div>'


def get_training_melody_icon(melody_key, melody_data):
    """Return a small visual marker for each training melody card."""
    key_text = str(melody_key).lower().strip()
    name_text = str(melody_data.get("name", "")).lower().strip()

    for label, icon in MELODY_ICONS.items():
        if label in key_text or label in name_text:
            return icon

    return "🎵"



PACK_SOURCES = {
    "pack:beginner_pack": {
        "file": "beginner_pack.json",
        "icon": "🌱",
        "fallback_name": "Beginner Pack",
        "short_description": "Strict, clean MelodyHub melodies for simpler outputs.",
    },
    "pack:public_domain_pack": {
        "file": "public_domain_pack.json",
        "icon": "📚",
        "fallback_name": "Public Domain Melodies Pack",
        "short_description": "A larger, looser public-domain-style MelodyHub pack.",
    },
}


@st.cache_data(show_spinner=False)
def load_melody_packs():
    """Load MelodyHub JSON packs and flatten them into pitch-rhythm training events."""
    loaded = {}

    for source_key, config in PACK_SOURCES.items():
        pack_path = DATA_DIR / config["file"]
        if not pack_path.exists():
            # Development fallback for this chat/exported test environment.
            local_fallback = APP_DIR / config["file"]
            if local_fallback.exists():
                pack_path = local_fallback
            else:
                continue

        with open(pack_path, "r", encoding="utf-8") as f:
            pack = json.load(f)

        melodies = pack.get("melodies", [])
        events = []
        for melody in melodies:
            for event in melody.get("events", []):
                if isinstance(event, (list, tuple)) and len(event) == 2:
                    pitch, rhythm = event
                    events.append((str(pitch), float(rhythm)))

        pitches = [pitch for pitch, _rhythm in events]
        rhythms = [rhythm for _pitch, rhythm in events]
        name = pack.get("name", config["fallback_name"])

        loaded[source_key] = {
            "key": source_key,
            "name": name,
            "description": pack.get("description", config["short_description"]),
            "icon": config["icon"],
            "source_type": "pack",
            "is_pack": True,
            "melody_count": int(pack.get("melody_count", len(melodies))),
            "note_count": len(events),
            "pitches": pitches,
            "rhythms": rhythms,
            "events": events,
        }

    return loaded


def get_single_training_sources():
    """Wrap the built-in training melodies in the same structure as JSON packs."""
    sources = {}
    for melody_key, melody_data in TRAINING_MELODIES.items():
        source_key = f"single:{melody_key}"
        pitches = list(melody_data.get("pitches", []))
        rhythms = list(melody_data.get("rhythms", []))
        sources[source_key] = {
            "key": source_key,
            "single_key": melody_key,
            "name": melody_data.get("name", melody_key),
            "description": melody_data.get("description", "Built-in demo melody."),
            "icon": get_training_melody_icon(melody_key, melody_data),
            "source_type": "single",
            "is_pack": False,
            "melody_count": 1,
            "note_count": len(pitches),
            "pitches": pitches,
            "rhythms": rhythms,
            "events": list(zip(pitches, rhythms)),
        }
    return sources


def normalize_training_source_key(source_key):
    """Support older session-state values like 'minuet' alongside new 'single:minuet'."""
    if source_key in TRAINING_MELODIES:
        return f"single:{source_key}"
    return source_key


def get_all_training_sources():
    """Return every available training source: JSON packs first, then single melodies."""
    sources = {}
    sources.update(load_melody_packs())
    sources.update(get_single_training_sources())
    return sources


def get_available_training_sources(model_name):
    """Apply model-specific training-source rules."""
    all_sources = get_all_training_sources()

    if model_name == "VariableMarkov":
        # Variable Markov is intentionally limited to the big packs and Minuet.
        return {
            key: value
            for key, value in all_sources.items()
            if value.get("source_type") == "pack" or value.get("single_key") == "minuet"
        }

    if model_name == "RuleBased":
        # Rule-Based stays on the original single built-in melodies only.
        return {
            key: value
            for key, value in all_sources.items()
            if value.get("source_type") == "single"
        }

    return all_sources


def get_selected_training_data():
    """Read the currently selected training source from session state."""
    source_key = normalize_training_source_key(st.session_state.selected_melody)
    return get_all_training_sources()[source_key]


def get_training_display_name(source_key):
    """Return a readable name for history/setup labels."""
    source_key = normalize_training_source_key(source_key)
    source = get_all_training_sources().get(source_key)
    return source["name"] if source else str(source_key)




# -----------------------------
# Shared UI polish
# -----------------------------
def inject_global_ui_css():
    """Global MelodyLab styling for pages, cards, and Streamlit buttons."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(124, 58, 237, 0.10), transparent 30%),
                radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 28%),
                linear-gradient(180deg, #fbfdff 0%, #f8fafc 100%);
        }
        .ml-hero {
            position: relative;
            overflow: hidden;
            padding: 2rem 2.1rem;
            border-radius: 30px;
            background: linear-gradient(135deg, #111827 0%, #312e81 52%, #7c3aed 100%);
            color: #ffffff;
            box-shadow: 0 24px 60px rgba(49, 46, 129, 0.26);
            margin-bottom: 1.3rem;
        }
        .ml-hero:after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -90px;
            top: -100px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.14);
        }
        .ml-brand-row {
            display: flex;
            align-items: center;
            gap: 1rem;
            position: relative;
            z-index: 1;
        }
        .ml-logo {
            width: 76px;
            height: 76px;
            border-radius: 24px;
            background: linear-gradient(135deg, #fef3c7, #f9a8d4 45%, #93c5fd);
            display: grid;
            place-items: center;
            color: #111827;
            font-weight: 950;
            font-size: 1.35rem;
            letter-spacing: -0.08em;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.55), 0 16px 34px rgba(0,0,0,0.25);
        }
        .ml-logo-img {
            width: 76px;
            height: 76px;
            border-radius: 26px;
            object-fit: cover;
            box-shadow: 0 16px 34px rgba(0,0,0,0.25);
            display: block;
        }
        .ml-title {
            font-size: clamp(2.5rem, 5vw, 4.9rem);
            line-height: 0.95;
            font-weight: 950;
            letter-spacing: -0.075em;
            margin: 0;
        }
        .ml-kicker {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.76rem;
            font-weight: 850;
            color: #c4b5fd;
            margin-bottom: 0.35rem;
        }
        .ml-subheading {
            position: relative;
            z-index: 1;
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 1.2rem;
            margin-bottom: 0.35rem;
        }
        .ml-subsubheading {
            position: relative;
            z-index: 1;
            max-width: 720px;
            color: #dbeafe;
            font-size: 1.02rem;
            line-height: 1.55;
        }
        .ml-section-title {
            font-size: 1.35rem;
            font-weight: 900;
            letter-spacing: -0.03em;
            color: #111827;
            margin: 0.75rem 0 0.5rem;
        }
        .ml-card {
            padding: 1.15rem 1.2rem;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
            margin-bottom: 0.8rem;
        }
        .ml-card-title {
            font-size: 1.05rem;
            font-weight: 900;
            color: #111827;
            margin-bottom: 0.2rem;
        }
        .ml-card-personality {
            display: inline-block;
            padding: 0.24rem 0.62rem;
            border-radius: 999px;
            background: #ede9fe;
            color: #5b21b6;
            font-size: 0.78rem;
            font-weight: 850;
            margin: 0.2rem 0 0.55rem 0;
        }
        .ml-card-text {
            color: #475569;
            font-size: 0.95rem;
            line-height: 1.4;
        }
        .ml-selected-pill {
            display: inline-block;
            padding: 0.48rem 0.85rem;
            border-radius: 999px;
            background: #eef2ff;
            color: #3730a3;
            font-weight: 850;
            border: 1px solid rgba(99, 102, 241, 0.20);
        }
        .ml-setup-card {
            padding: 1.1rem 1.2rem;
            border-radius: 22px;
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.07);
        }
        .ml-setup-label {
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .ml-setup-value {
            color: #111827;
            font-size: 1.12rem;
            font-weight: 900;
            margin-top: 0.25rem;
        }
        div.stButton > button {
            min-height: 3.2rem;
            border-radius: 16px;
            border: 1px solid rgba(99, 102, 241, 0.22);
            background: #ffffff;
            color: #1f2937;
            font-weight: 900;
            letter-spacing: -0.01em;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.07);
            transition: all 0.16s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            border-color: rgba(124, 58, 237, 0.45);
            box-shadow: 0 16px 30px rgba(79, 70, 229, 0.16);
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            border: 0;
            min-height: 3.9rem;
            font-size: 1.12rem;
            font-weight: 950;
            box-shadow: 0 18px 38px rgba(79, 70, 229, 0.25);
        }
        div[data-testid="stExpander"] {
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.28);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_hero():
    """Render the MelodyLab landing header with a custom logo-style mark."""
    logo_markup = get_logo_markup()
    st.markdown(
        f"""
        <div class="ml-hero">
            <div class="ml-brand-row">
                {logo_markup}
                <div>
                    <div class="ml-kicker">melody generator lab</div>
                    <h1 class="ml-title">MelodyLab</h1>
                </div>
            </div>
            <div class="ml-subheading">Can math make music?</div>
            <div class="ml-subsubheading">
                Choose a model, train it on a melody, generate a new tune, listen to it, then inspect how every note was chosen.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


MODEL_UI = {
    "Random": {
        "name": "Random",
        "personality": "The Dice Roller",
        "meaning": "Chooses full pitch-rhythm note events by chance from the selected melody.",
        "button": "Choose Random",
    },
    "WeightedRandom": {
        "name": "Weighted Random",
        "personality": "The Loaded Dice Roller",
        "meaning": "Chooses note events by chance, but events that appeared more often in the training melody are more likely.",
        "button": "Choose Weighted Random",
    },
    "Markov1": {
        "name": "First-Order Markov",
        "personality": "The One-Note Listener",
        "meaning": "Looks at the previous note event before choosing what comes next.",
        "button": "Choose First-Order Markov",
    },
    "Markov2": {
        "name": "Second-Order Markov",
        "personality": "The Pattern Imitator",
        "meaning": "Looks at the previous two note events, so it can imitate short patterns.",
        "button": "Choose Second-Order Markov",
    },
    "VariableMarkov": {
        "name": "Variable Markov",
        "personality": "The Flexible Pattern Listener",
        "meaning": "Tries to use up to five previous note events, then backs off to shorter memories when a pattern is missing or too rare.",
        "button": "Choose Variable Markov",
    },
    "RuleBased": {
        "name": "Rule-Based",
        "personality": "The Music Theory Student",
        "meaning": "Follows encoded music-theory rules instead of transition probabilities.",
        "button": "Choose Rule-Based",
    },
}


def get_model_display_name(model_name):
    info = MODEL_UI.get(model_name)
    if not info:
        return str(model_name)
    return f'{info["name"]} ({info["personality"]})'


def render_page_header(title, subtitle=""):
    st.markdown(f'<div class="ml-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="ml-card-text" style="margin-bottom:1rem;">{subtitle}</div>', unsafe_allow_html=True)


# -----------------------------
# Melody history + comparison
# -----------------------------
def score_to_percent(score):
    """Convert a 0-1 score into a 0-100 integer."""
    return int(round(clamp_score(score) * 100))


def format_melody_sequence(melody):
    """Return a compact pitch-duration sequence for history cards."""
    return " · ".join(f"{pitch}({duration})" for pitch, duration in melody)


def get_score_value(results, key, inverse=False):
    """Safely read a scorecard value and optionally invert it."""
    value = clamp_score(results.get(key, 0.0))
    if inverse:
        value = 1.0 - value
    return value


def get_saved_melody_label(index, entry):
    """Label used in dropdowns and history cards."""
    overall = score_to_percent(entry["results"].get("final_score", 0.0))
    return (
        f"Melody {index + 1}: {entry['model_display']} · "
        f"{entry['training_melody_name']} · {overall}/100"
    )


def save_current_melody_to_history(melody, scorecard_results):
    """Auto-save a generated melody so the user can replay and compare it later."""
    selected_training_data = get_selected_training_data()

    history_entry = {
        "model": st.session_state.selected_model,
        "model_display": get_model_display_name(st.session_state.selected_model),
        "training_melody": normalize_training_source_key(st.session_state.selected_melody),
        "training_melody_name": selected_training_data["name"],
        "rule_mode": st.session_state.rule_mode,
        "length": len(melody),
        "melody": list(melody),
        "results": dict(scorecard_results),
    }

    st.session_state.melody_history.append(history_entry)
    return len(st.session_state.melody_history)


def show_saved_melody_scorecard(entry):
    """Render a compact scorecard for one saved history item."""
    results = entry["results"]

    smoothness = get_score_value(results, "interval_smoothness")
    jumps = get_score_value(results, "interval_smoothness", inverse=True)
    patterns = get_score_value(results, "motif_repetition")
    rhythm = get_score_value(results, "rhythmic_variety")
    overall = get_score_value(results, "final_score")

    st.write(f"**Model:** {entry['model_display']}")
    st.write(f"**Training melody:** {entry['training_melody_name']}")
    if entry.get("rule_mode"):
        st.write(f"**Rule mode:** {entry['rule_mode'].title()}")
    st.write(f"**Length:** {entry['length']} notes")

    st.markdown(
        f"""
        | Metric | Score |
        |---|---:|
        | Moves smoothly | {score_to_percent(smoothness)} |
        | Jumps around | {score_to_percent(jumps)} |
        | Repeats ideas | {score_to_percent(patterns)} |
                | Varied beats | {score_to_percent(rhythm)} |
        | Overall | {score_to_percent(overall)} |
        """
    )


def page_history():
    """Show all melodies generated during the current app session."""
    inject_global_ui_css()
    render_page_header(
        "Melody history",
        "Every generated melody is saved here during this app session so you can replay it and compare outputs.",
    )

    history = st.session_state.melody_history

    if not history:
        st.info("No melodies saved yet. Generate a melody first.")
        if st.button("← Back to model selection", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        return

    top_col1, top_col2, top_col3 = st.columns([1, 1, 2])
    with top_col1:
        if st.button("← Back to Results", use_container_width=True):
            st.session_state.page = "results"
            st.rerun()
    with top_col2:
        if st.button("Clear History", use_container_width=True):
            st.session_state.melody_history = []
            st.session_state.saved_history_index = None
            st.session_state.page = "home"
            st.rerun()

    st.divider()

    compare_col, spacer_col = st.columns([1, 3])
    with compare_col:
        if st.button("⚖️ Compare Melodies", use_container_width=True, disabled=len(history) < 2):
            st.session_state.page = "compare"
            st.rerun()

    st.divider()

    for index, entry in enumerate(reversed(history)):
        original_index = len(history) - 1 - index
        label = get_saved_melody_label(original_index, entry)
        overall = score_to_percent(entry["results"].get("final_score", 0.0))

        st.markdown(
            f"""
            <div class="ml-card">
                <div class="ml-card-title">{label}</div>
                <div class="ml-card-text">
                    {entry['length']} notes generated from <b>{entry['training_melody_name']}</b>
                    {f" · {entry['rule_mode'].title()} mode" if entry.get('rule_mode') else ""}
                </div>
                <div class="ml-card-personality">Overall score: {overall}/100</div>
                <div class="ml-card-text">{format_melody_sequence(entry['melody'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(f"Open Melody {original_index + 1}", expanded=False):
            st.write("#### Play melody")
            render_play_melody_button(melody_to_wav(entry["melody"], tempo=120))
            st.write("#### Scorecard")
            show_saved_melody_scorecard(entry)

        st.divider()


def page_compare():
    """Compare two saved melodies side-by-side."""
    inject_global_ui_css()

    if st.button("← Back to model selection", key="compare_back_home"):
        st.session_state.page = "home"
        st.rerun()

    render_page_header(
        "Compare two melodies",
        "Select any two saved melodies, compare their scorecards, and play them side by side.",
    )

    history = st.session_state.melody_history

    if len(history) < 2:
        st.warning("Generate at least two melodies before comparing.")
        return

    labels = [get_saved_melody_label(i, entry) for i, entry in enumerate(history)]

    col1, col2 = st.columns(2)
    with col1:
        first_label = st.selectbox("Melody A", labels, index=max(0, len(labels) - 2), key="compare_a")
    with col2:
        second_label = st.selectbox("Melody B", labels, index=len(labels) - 1, key="compare_b")

    index_a = labels.index(first_label)
    index_b = labels.index(second_label)

    if index_a == index_b:
        st.warning("Choose two different melodies to compare.")
        return

    melody_a = history[index_a]
    melody_b = history[index_b]

    st.divider()

    st.write("### Listen first")
    play_col1, play_col2 = st.columns(2)
    with play_col1:
        st.write("#### Melody A")
        st.caption(get_saved_melody_label(index_a, melody_a))
        render_play_melody_button(melody_to_wav(melody_a["melody"], tempo=120))
    with play_col2:
        st.write("#### Melody B")
        st.caption(get_saved_melody_label(index_b, melody_b))
        render_play_melody_button(melody_to_wav(melody_b["melody"], tempo=120))

    st.divider()


    comparison_rows = [
        ("Model", melody_a["model_display"], melody_b["model_display"]),
        ("Training Melody", melody_a["training_melody_name"], melody_b["training_melody_name"]),
        ("Rule Mode", melody_a["rule_mode"].title() if melody_a.get("rule_mode") else "—", melody_b["rule_mode"].title() if melody_b.get("rule_mode") else "—"),
        ("Length", f"{melody_a['length']} notes", f"{melody_b['length']} notes"),
        ("Moves smoothly", score_to_percent(get_score_value(melody_a["results"], "interval_smoothness")), score_to_percent(get_score_value(melody_b["results"], "interval_smoothness"))),
        ("Jumps around", score_to_percent(get_score_value(melody_a["results"], "interval_smoothness", inverse=True)), score_to_percent(get_score_value(melody_b["results"], "interval_smoothness", inverse=True))),
        ("Repeats ideas", score_to_percent(get_score_value(melody_a["results"], "motif_repetition")), score_to_percent(get_score_value(melody_b["results"], "motif_repetition"))),
        ("Varied beats", score_to_percent(get_score_value(melody_a["results"], "rhythmic_variety")), score_to_percent(get_score_value(melody_b["results"], "rhythmic_variety"))),
        ("Overall", score_to_percent(get_score_value(melody_a["results"], "final_score")), score_to_percent(get_score_value(melody_b["results"], "final_score"))),
    ]

    st.write("### Score comparison")
    table_md = "| Feature | Melody A | Melody B |\n|---|---:|---:|\n"
    for feature, value_a, value_b in comparison_rows:
        table_md += f"| {feature} | {value_a} | {value_b} |\n"
    st.markdown(table_md)

    st.write("### Melody sequences")
    seq_col1, seq_col2 = st.columns(2)
    with seq_col1:
        st.write("**Melody A**")
        st.caption(format_melody_sequence(melody_a["melody"]))
    with seq_col2:
        st.write("**Melody B**")
        st.caption(format_melody_sequence(melody_b["melody"]))



# Page: HOME - Model Selection
def page_home():
    inject_global_ui_css()
    render_brand_hero()
    render_page_header("Choose a model", "Each model has a different composing personality. Start with one and compare the results later.")

    cols = st.columns(6)
    model_keys = ["Random", "WeightedRandom", "RuleBased", "Markov1", "Markov2", "VariableMarkov"]
    for col, model_key in zip(cols, model_keys):
        info = MODEL_UI[model_key]
        with col:
            st.markdown(
                f"""
                <div class="ml-card">
                    <div class="ml-card-title">{info['name']}</div>
                    <div class="ml-card-personality">{info['personality']}</div>
                    <div class="ml-card-text">{info['meaning']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(info["button"], use_container_width=True, key=f"btn_{model_key}"):
                st.session_state.selected_model = model_key
                st.session_state.model_instance = None
                if model_key == "RuleBased":
                    st.session_state.page = "rule_mode"
                else:
                    st.session_state.page = "training_melody"
                st.rerun()

    if st.session_state.melody_history:
        st.divider()
        history_col, spacer_col = st.columns([1, 3])
        with history_col:
            if st.button("📜 Melody History", use_container_width=True):
                st.session_state.page = "history"
                st.rerun()

# Page: RULE-BASED MODE Selection
def page_rule_mode():
    inject_global_ui_css()
    render_page_header(
        "Choose rule-based mode",
        "Strict mode always chooses the highest-scoring rule-approved note. Creative mode chooses from the top few options for more variety.",
    )
    st.markdown(
        f'Selected Model: <span class="ml-selected-pill">{get_model_display_name("RuleBased")}</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="ml-card">
                <div class="ml-card-title">Strict Mode</div>
                <div class="ml-card-personality">More predictable</div>
                <div class="ml-card-text">Always chooses the highest-scoring note according to the encoded music-theory rules.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Choose Strict Mode", use_container_width=True, key="btn_rule_strict"):
            st.session_state.rule_mode = "strict"
            st.session_state.page = "training_melody"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="ml-card">
                <div class="ml-card-title">Creative Mode</div>
                <div class="ml-card-personality">More variety</div>
                <div class="ml-card-text">Chooses from the top few rule-approved notes, so the output is less repetitive.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Choose Creative Mode", use_container_width=True, key="btn_rule_creative"):
            st.session_state.rule_mode = "creative"
            st.session_state.page = "training_melody"
            st.rerun()

    st.divider()
    if st.button("← Back to model selection", use_container_width=True):
        st.session_state.page = "home"
        st.session_state.selected_model = None
        st.session_state.rule_mode = None
        st.rerun()


# Page: TRAINING MELODY Selection
def page_training_melody():
    inject_global_ui_css()
    render_page_header(
        "Choose training data",
        "Train on one large MelodyHub pack, or keep things small with a single built-in melody.",
    )
    st.markdown(
        f'Selected Model: <span class="ml-selected-pill">{get_model_display_name(st.session_state.selected_model)}</span>',
        unsafe_allow_html=True,
    )
    if st.session_state.selected_model == "RuleBased":
        st.markdown(
            f'Rule-Based Mode: <span class="ml-selected-pill">{(st.session_state.rule_mode or "strict").title()} Mode</span>',
            unsafe_allow_html=True,
        )
        st.caption("Rule-Based uses the original built-in single melodies only.")
    if st.session_state.selected_model == "VariableMarkov":
        st.caption("Variable Markov is limited to MelodyHub packs and Minuet, so it has enough repeated patterns to learn from.")
    st.divider()

    available_sources = get_available_training_sources(st.session_state.selected_model)
    pack_options = {key: data for key, data in available_sources.items() if data.get("source_type") == "pack"}
    single_options = {key: data for key, data in available_sources.items() if data.get("source_type") == "single"}

    if pack_options:
        st.write("### Melody packs")
        pack_cols = st.columns(min(2, len(pack_options)))
        for idx, (source_key, source_data) in enumerate(pack_options.items()):
            with pack_cols[idx % len(pack_cols)]:
                st.markdown(
                    f"""
                    <div class="ml-card">
                        <div class="ml-card-title">{source_data['icon']} {source_data['name']}</div>
                        <div class="ml-card-personality">{source_data['melody_count']} melodies · {source_data['note_count']} note-events</div>
                        <div class="ml-card-text">{source_data['description']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Train on {source_data['icon']} {source_data['name']}", use_container_width=True, key=f"btn_{source_key}"):
                    st.session_state.selected_melody = source_key
                    st.session_state.model_instance = None
                    st.session_state.generated_melody = None
                    st.session_state.scorecard_results = None
                    st.session_state.page = "generate"
                    st.rerun()

    if single_options:
        st.write("### Single melodies")
        single_cols = st.columns(3)
        for idx, (source_key, source_data) in enumerate(single_options.items()):
            with single_cols[idx % 3]:
                st.markdown(
                    f"""
                    <div class="ml-card">
                        <div class="ml-card-title">{source_data['icon']} {source_data['name']}</div>
                        <div class="ml-card-text">{source_data['note_count']} training notes</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Train on {source_data['icon']} {source_data['name']}", use_container_width=True, key=f"btn_{source_key}"):
                    st.session_state.selected_melody = source_key
                    st.session_state.model_instance = None
                    st.session_state.generated_melody = None
                    st.session_state.scorecard_results = None
                    st.session_state.page = "generate"
                    st.rerun()

    if not available_sources:
        st.error("No training data was found. Check that the JSON files are inside data/processed.")

    st.divider()
    if st.button("← Back to model selection", use_container_width=True):
        st.session_state.page = "home"
        st.session_state.selected_model = None
        st.rerun()

# Page: GENERATE Melody
def page_generate():
    inject_global_ui_css()

    if st.button("← Back to model selection", key="generate_back_home"):
        st.session_state.page = "home"
        st.session_state.selected_model = None
        st.session_state.selected_melody = None
        st.session_state.model_instance = None
        st.session_state.generated_melody = None
        st.session_state.scorecard_results = None
        st.session_state.learning_step = 0
        st.session_state.rule_mode = None
        st.rerun()

    render_page_header("Generate melody", "Review your setup, then create a melody from the selected model.")

    selected_training_data = get_selected_training_data()
    melody_display = selected_training_data["name"]

    st.session_state.melody_length = max(
        MIN_MELODY_LENGTH,
        min(MAX_MELODY_LENGTH, int(st.session_state.melody_length)),
    )

    melody_length = st.slider(
        "Melody length",
        min_value=MIN_MELODY_LENGTH,
        max_value=MAX_MELODY_LENGTH,
        value=st.session_state.melody_length,
        step=1,
        help="Choose how many notes the model should generate.",
    )
    st.session_state.melody_length = melody_length

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="ml-setup-card"><div class="ml-setup-label">Model</div><div class="ml-setup-value">{get_model_display_name(st.session_state.selected_model)}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="ml-setup-card"><div class="ml-setup-label">Training Melody</div><div class="ml-setup-value">{melody_display}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="ml-setup-card"><div class="ml-setup-label">Length</div><div class="ml-setup-value">{st.session_state.melody_length} notes</div></div>', unsafe_allow_html=True)

    if st.session_state.selected_model == "RuleBased":
        st.markdown(
            f'<div style="margin-top:0.9rem;"><span class="ml-selected-pill">Rule-Based Mode: {(st.session_state.rule_mode or "strict").title()}</span></div>',
            unsafe_allow_html=True,
        )
    
    st.divider()
    
    # Initialize model instance if needed - USE SELECTED MELODY
    if st.session_state.model_instance is None:
        selected_melody_data = get_selected_training_data()
        pitches = selected_melody_data["pitches"]
        rhythms = selected_melody_data["rhythms"]
        
        if st.session_state.selected_model == "Random":
            st.session_state.model_instance = RandomMelodyGenerator(
                pitches=pitches,
                rhythms=rhythms
            )
        elif st.session_state.selected_model == "WeightedRandom":
            st.session_state.model_instance = WeightedRandomMelodyGenerator(
                pitches=pitches,
                rhythms=rhythms
            )
        elif st.session_state.selected_model == "Markov1":
            st.session_state.model_instance = FirstOrderMarkovGenerator(
                pitches=pitches,
                rhythms=rhythms
            )
        elif st.session_state.selected_model == "Markov2":
            st.session_state.model_instance = SecondOrderMarkovGenerator(
                pitches=pitches,
                rhythms=rhythms
            )
        elif st.session_state.selected_model == "VariableMarkov":
            st.session_state.model_instance = VariableMarkovGenerator(
                pitches=pitches,
                rhythms=rhythms,
                max_order=5
            )
        elif st.session_state.selected_model == "RuleBased":
            st.session_state.model_instance = RuleBasedMelodyGenerator(
                training_melody_key=selected_melody_data.get("single_key", "minuet"),
                mode=st.session_state.rule_mode or "strict"
            )
    
    # Generate Melody Button
    if st.button("✨ Generate Melody →", use_container_width=True, key="btn_generate", type="primary"):
        melody = st.session_state.model_instance.generate_melody(length=st.session_state.melody_length)
        st.session_state.generated_melody = melody
        st.session_state.learning_step = 0
        st.session_state.play_learning_note = False
        
        # Evaluate melody with scorecard
        selected_melody_data = get_selected_training_data()
        training_melody = list(zip(selected_melody_data["pitches"], selected_melody_data["rhythms"]))
        d_max = infer_duration_vocab_size([training_melody])
        scorecard_results = evaluate_sequence(melody, d_max)
        st.session_state.scorecard_results = scorecard_results
        st.session_state.saved_history_index = save_current_melody_to_history(melody, scorecard_results) - 1
        
        # Move to results page
        st.session_state.page = "results"
        st.rerun()
    
# Helper: Generate visual bar
def create_bar(score, max_segments=10):
    """Create a visual bar for score (0-1 scale)."""
    filled = round(score * max_segments)
    empty = max_segments - filled
    return "█" * filled + "░" * empty

# Helper: Get category label
def get_category(score, metric_type="standard"):
    """Categorize score based on thresholds."""
    if metric_type == "rhythm":
        if score >= 0.7:
            return "Varied"
        elif score >= 0.4:
            return "Balanced"
        else:
            return "Repetitive"
    else:  # standard
        if score >= 0.7:
            return "High"
        elif score >= 0.4:
            return "Med"
        else:
            return "Low"

def clamp_score(score):
    """Keep all score values safely inside the 0-1 display range."""
    try:
        return max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        return 0.0

def get_score_color(score):
    """Return a color for score bands."""
    score = clamp_score(score)
    if score >= 0.7:
        return "#22c55e"  # green
    if score >= 0.4:
        return "#f59e0b"  # amber
    return "#ef4444"      # red

def inject_scorecard_css():
    """Style the scorecard so it looks like part of the MelodyLab UI."""
    st.markdown(
        """
        <style>
        .score-hero {
            padding: 1.35rem 1.5rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #fff7ed 0%, #eef2ff 55%, #fdf2f8 100%);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
        }
        .score-hero-title {
            font-size: 1.05rem;
            font-weight: 750;
            color: #334155;
            margin-bottom: 0.35rem;
        }
        .score-hero-number {
            font-size: 3rem;
            line-height: 1;
            font-weight: 850;
            color: #111827;
            letter-spacing: -0.05em;
        }
        .score-hero-subtitle {
            color: #64748b;
            font-size: 0.95rem;
            margin-top: 0.35rem;
        }
        .score-card {
            padding: 1rem 1rem 0.9rem 1rem;
            border-radius: 20px;
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
            min-height: 150px;
            margin-bottom: 0.9rem;
        }
        .score-card-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.75rem;
        }
        .score-card-title {
            font-size: 1rem;
            font-weight: 750;
            color: #1f2937;
        }
        .score-card-help {
            font-size: 0.82rem;
            color: #64748b;
            margin-top: 0.2rem;
        }
        .score-pill {
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 750;
            white-space: nowrap;
        }
        .score-value-row {
            display: flex;
            align-items: baseline;
            gap: 0.35rem;
            margin-top: 0.9rem;
            margin-bottom: 0.55rem;
        }
        .score-value {
            font-size: 1.65rem;
            font-weight: 850;
            color: #111827;
            letter-spacing: -0.03em;
        }
        .score-scale {
            color: #94a3b8;
            font-size: 0.9rem;
            font-weight: 650;
        }
        .score-track {
            height: 0.72rem;
            width: 100%;
            overflow: hidden;
            border-radius: 999px;
            background: #e5e7eb;
        }
        .score-fill {
            height: 100%;
            border-radius: 999px;
        }
        .score-note {
            color: #64748b;
            font-size: 0.9rem;
            margin-top: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def metric_card(title, score, category, help_text, icon="🎵"):
    """Render one visual scorecard tile."""
    score = clamp_score(score)
    percent = int(round(score * 100))
    color = get_score_color(score)
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-card-top">
                <div>
                    <div class="score-card-title">{icon} {title}</div>
                    <div class="score-card-help">{help_text}</div>
                </div>
                <div class="score-pill" style="background:{color};">{category}</div>
            </div>
            <div class="score-value-row">
                <div class="score-value">{percent}</div>
                <div class="score-scale">/100</div>
            </div>
            <div class="score-track">
                <div class="score-fill" style="width:{percent}%; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Helper: Convert generated melody into playable WAV audio
def note_to_frequency(note):
    """Convert a note name like G4 or F#5 into frequency in Hz."""
    note = str(note).strip()

    # Treat rest tokens as silence.
    if note.upper().startswith("REST"):
        return None

    # Support both sharps and flats.
    semitone_map = {
        "C": 0, "C#": 1, "DB": 1,
        "D": 2, "D#": 3, "EB": 3,
        "E": 4,
        "F": 5, "F#": 6, "GB": 6,
        "G": 7, "G#": 8, "AB": 8,
        "A": 9, "A#": 10, "BB": 10,
        "B": 11,
    }

    if len(note) < 2:
        return 440.0

    if len(note) >= 3 and note[1] in ["#", "b", "-"]:
        accidental = "B" if note[1] in ["b", "-"] else "#"
        pitch_class = note[0].upper() + accidental
        octave_text = note[2:]
    else:
        pitch_class = note[0].upper()
        octave_text = note[1:]

    try:
        octave = int(octave_text)
    except ValueError:
        return 440.0

    if pitch_class not in semitone_map:
        return 440.0

    midi_number = 12 * (octave + 1) + semitone_map[pitch_class]
    return 440.0 * (2 ** ((midi_number - 69) / 12))


def melody_to_wav(melody, tempo=120):
    """Create an in-memory WAV file from [(pitch, duration), ...]."""
    sample_rate = 44100
    beat_seconds = 60 / tempo
    audio_parts = []

    for pitch, duration in melody:
        seconds = max(0.05, float(duration) * beat_seconds)
        t = np.linspace(0, seconds, int(sample_rate * seconds), False)
        freq = note_to_frequency(pitch)

        if freq is None:
            note_audio = np.zeros_like(t)
        else:
            note_audio = 0.3 * np.sin(2 * np.pi * freq * t)

            # Fade in/out to prevent clicking between notes.
            fade_len = min(500, len(note_audio) // 10)
            if fade_len > 0:
                note_audio[:fade_len] *= np.linspace(0, 1, fade_len)
                note_audio[-fade_len:] *= np.linspace(1, 0, fade_len)

        audio_parts.append(note_audio)

    if audio_parts:
        audio = np.concatenate(audio_parts)
    else:
        audio = np.array([], dtype=np.float32)

    audio_int16 = np.int16(audio * 32767)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    wav_buffer.seek(0)
    return wav_buffer.getvalue()


def render_play_melody_button(audio_bytes):
    """Render a prominent button that plays the generated melody."""
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    st.iframe(
        f"""
        <style>
            .melody-player {{
                display: flex;
                align-items: center;
                gap: 0.85rem;
                padding: 0.9rem 1rem;
                border: 1px solid rgba(148, 163, 184, 0.28);
                border-radius: 8px;
                background: #ffffff;
                box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
                font-family: "Source Sans Pro", sans-serif;
            }}
            .play-melody-button {{
                border: 0;
                border-radius: 8px;
                padding: 0.7rem 1rem;
                background: #2563eb;
                color: #ffffff;
                font-size: 0.95rem;
                font-weight: 700;
                cursor: pointer;
                min-width: 145px;
            }}
            .play-melody-button:hover {{
                background: #1d4ed8;
            }}
            .play-melody-status {{
                color: #475569;
                font-size: 0.92rem;
                line-height: 1.25;
            }}
        </style>
        <div class="melody-player">
            <button class="play-melody-button" type="button" id="play-melody">
                ▶ Play Melody
            </button>
            <span class="play-melody-status" id="play-status">Ready to play the generated melody.</span>
            <audio id="melody-audio" preload="auto">
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            </audio>
        </div>
        <script>
            const button = document.getElementById("play-melody");
            const status = document.getElementById("play-status");
            const audio = document.getElementById("melody-audio");

            button.addEventListener("click", async () => {{
                audio.currentTime = 0;
                try {{
                    await audio.play();
                    status.textContent = "Playing melody...";
                }} catch (error) {{
                    status.textContent = "Your browser blocked playback. Try clicking the button again.";
                }}
            }});

            audio.addEventListener("ended", () => {{
                status.textContent = "Ready to play the generated melody.";
            }});
        </script>
        """,
        height=90,
    )


def autoplay_single_note(pitch, duration, tempo=120):
    """
    Autoplay the currently revealed note in Learning Mode.

    This is triggered after the user clicks Think Next Note and Streamlit reruns
    the page. Browser autoplay rules can still vary, but because the rerun follows
    a user click, this usually works without needing a separate play button.
    """
    audio_bytes = melody_to_wav([(pitch, duration)], tempo=tempo)
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    st.iframe(
        f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
        """,
        height=0,
    )


# Helper: Learning Mode visuals and explanations
def inject_learning_css():
    """Style the note-by-note learning mode."""
    st.markdown(
        """
        <style>
        .learning-hero {
            padding: 1.25rem 1.35rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #fff7ed 0%, #eef2ff 55%, #fdf2f8 100%);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
        }
        .learning-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: #334155;
            margin-bottom: 0.35rem;
        }
        .melody-builder {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            border-radius: 22px;
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            margin: 0.75rem 0 1rem 0;
        }
        .note-chip {
            min-width: 58px;
            padding: 0.55rem 0.65rem;
            border-radius: 16px;
            text-align: center;
            font-weight: 850;
            color: #1f2937;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }
        .note-chip.current {
            color: #ffffff;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            border: 1px solid rgba(37, 99, 235, 0.25);
            transform: translateY(-2px);
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22);
        }
        .note-chip.future {
            color: #94a3b8;
            background: #f1f5f9;
            border: 1px dashed #cbd5e1;
        }
        .learning-card {
            padding: 1rem 1.05rem;
            border-radius: 20px;
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.9rem;
            min-height: 135px;
        }
        .learning-card-title {
            font-size: 1rem;
            font-weight: 800;
            color: #1f2937;
            margin-bottom: 0.45rem;
        }
        .learning-card-body {
            color: #475569;
            font-size: 0.95rem;
            line-height: 1.45;
        }
        .choice-row {
            margin-bottom: 0.65rem;
        }
        .choice-label {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            color: #334155;
            font-size: 0.92rem;
            font-weight: 750;
            margin-bottom: 0.25rem;
        }
        .choice-track {
            width: 100%;
            height: 0.72rem;
            border-radius: 999px;
            overflow: hidden;
            background: #e5e7eb;
        }
        .choice-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #2563eb, #7c3aed);
        }
        .selected-note {
            display: inline-block;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: #dcfce7;
            color: #166534;
            font-weight: 850;
            margin-top: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_model_friendly_name(model_name):
    """Return a user-friendly model label."""
    return get_model_display_name(model_name)


def format_event(event):
    """Return a readable label for a pitch-rhythm event."""
    if isinstance(event, (list, tuple)) and len(event) == 2:
        pitch, duration = event
        beat_word = "beat" if float(duration) == 1.0 else "beats"
        return f"{pitch} ({duration} {beat_word})"
    return str(event)


def normalize_choices(choices):
    """Sort and clean probability choices while preserving optional raw scores."""
    cleaned = []
    for choice in choices:
        if len(choice) == 3:
            item, prob, raw_score = choice
        else:
            item, prob = choice
            raw_score = None

        try:
            p = float(prob)
        except (TypeError, ValueError):
            p = 0.0

        p = max(0.0, min(1.0, p))

        if raw_score is None:
            cleaned.append((item, p))
        else:
            cleaned.append((item, p, raw_score))

    return sorted(cleaned, key=lambda item: item[1], reverse=True)


def get_learning_step_info(model_name, model_instance, generated_events, generated_pitches, step):
    """
    Explain how the current event could have been selected.

    For probabilistic models, this reads event pools/chains from the model instance.
    For the rule-based model, it keeps using the pitch-based rule trace.
    """
    selected_event = tuple(generated_events[step])
    selected_pitch = generated_pitches[step]

    if model_name == "Random":
        event_pool = getattr(model_instance, "event_pool", sorted(set(getattr(model_instance, "events", generated_events))))
        probability = 1.0 / len(event_pool) if event_pool else 0.0
        return {
            "memory": "No memory. This model chooses each full note event independently.",
            "choices": normalize_choices([(event, probability) for event in event_pool]),
            "selected": selected_event,
            "selected_label": format_event(selected_event),
            "explanation": f"{format_event(selected_event)} was selected by chance from the available training note-events.",
            "fallback": False,
            "choice_kind": "event_probability",
        }

    if model_name == "WeightedRandom":
        event_pool = getattr(model_instance, "event_pool", [])
        event_weights = getattr(model_instance, "event_weights", [])
        choices = list(zip(event_pool, event_weights))

        return {
            "memory": (
                "No sequence memory. This model does not look at the previous note. "
                "It uses how often each full pitch-rhythm event appeared in the training melody as its probability weight."
            ),
            "choices": normalize_choices(choices),
            "selected": selected_event,
            "selected_label": format_event(selected_event),
            "explanation": (
                f"{format_event(selected_event)} was selected by weighted chance. "
                "Note-events that appeared more often in the training melody had a higher probability."
            ),
            "fallback": False,
            "choice_kind": "event_probability",
        }

    if model_name == "Markov1":
        if step == 0:
            return {
                "memory": "No previous note event yet.",
                "choices": [],
                "selected": selected_event,
                "selected_label": format_event(selected_event),
                "explanation": f"The melody starts with {format_event(selected_event)}. The first event is the starting point before one-event memory begins.",
                "fallback": False,
                "choice_kind": "event_probability",
            }

        previous_event = tuple(generated_events[step - 1])
        chain = getattr(model_instance, "event_chain", {})
        choices = chain.get(previous_event, [])

        if choices:
            explanation = (
                f"The model looked at the previous note-event, {format_event(previous_event)}, "
                "and used the training melody's event transition probabilities."
            )
            fallback = False
        else:
            keys = sorted(chain.keys())
            probability = 1.0 / len(keys) if keys else 0.0
            choices = [(event, probability) for event in keys]
            explanation = (
                f"The model had no stored next-event options after {format_event(previous_event)}, "
                "so it restarted from a valid training event."
            )
            fallback = True

        return {
            "memory": f"Previous note-event: {format_event(previous_event)}",
            "choices": normalize_choices(choices),
            "selected": selected_event,
            "selected_label": format_event(selected_event),
            "explanation": explanation,
            "fallback": fallback,
            "choice_kind": "event_probability",
        }

    if model_name == "Markov2":
        if step == 0:
            return {
                "memory": "No two-event memory yet.",
                "choices": [],
                "selected": selected_event,
                "selected_label": format_event(selected_event),
                "explanation": f"The melody starts with {format_event(selected_event)}. The second-order model first needs a two-event starting pair.",
                "fallback": False,
                "choice_kind": "event_probability",
            }

        if step == 1:
            previous_event = tuple(generated_events[step - 1])
            return {
                "memory": f"Starting pair being formed: {format_event(previous_event)} → {format_event(selected_event)}",
                "choices": [],
                "selected": selected_event,
                "selected_label": format_event(selected_event),
                "explanation": f"{format_event(selected_event)} completes the starting pair. After this, the model can use two-event memory.",
                "fallback": False,
                "choice_kind": "event_probability",
            }

        previous_pair = (tuple(generated_events[step - 2]), tuple(generated_events[step - 1]))
        second_order_chain = getattr(model_instance, "event_chain_2", {})
        first_order_chain = getattr(model_instance, "event_chain_1", {})
        second_order_choices = second_order_chain.get(previous_pair, [])

        if second_order_choices:
            choices = second_order_choices
            explanation = (
                f"The model looked at the two-event pattern {format_event(previous_pair[0])} → {format_event(previous_pair[1])} "
                "and used the training melody's second-order transition probabilities."
            )
            fallback = False
        else:
            last_event = previous_pair[1]
            first_order_choices = first_order_chain.get(last_event, [])
            fallback = True

            if first_order_choices:
                choices = first_order_choices
                explanation = (
                    f"The exact two-event pattern {format_event(previous_pair[0])} → {format_event(previous_pair[1])} was not stored, "
                    f"so the model backed off to one-event memory after {format_event(last_event)}."
                )
            else:
                event_pool = getattr(model_instance, "event_pool", [])
                event_weights = getattr(model_instance, "event_weights", [])
                choices = list(zip(event_pool, event_weights))
                explanation = (
                    f"The exact two-event pattern and the one-event fallback after {format_event(last_event)} were unavailable, "
                    "so the model used weighted-random training events as its final fallback."
                )

        return {
            "memory": f"Previous two note-events: {format_event(previous_pair[0])} → {format_event(previous_pair[1])}",
            "choices": normalize_choices(choices),
            "selected": selected_event,
            "selected_label": format_event(selected_event),
            "explanation": explanation,
            "fallback": fallback,
            "choice_kind": "event_probability",
        }

    if model_name == "VariableMarkov":
        trace_item = model_instance.get_trace_for_step(step) if hasattr(model_instance, "get_trace_for_step") else None

        if not trace_item:
            return {
                "memory": "Starting note-event copied from a random training fragment. Variable-order memory begins after the starting fragment.",
                "choices": [],
                "selected": selected_event,
                "selected_label": format_event(selected_event),
                "explanation": f"The melody starts with {format_event(selected_event)}. The model uses this starting fragment before it begins choosing new events with variable-order memory.",
                "fallback": False,
                "choice_kind": "event_probability",
            }

        attempts = trace_item.get("attempts", [])
        attempt_lines = []
        for attempt in attempts:
            order = attempt.get("order", "?")
            status = attempt.get("status", "checked")
            total_count = attempt.get("total_count", 0)
            min_count = attempt.get("min_count", 1)
            context_events = [
                (item.get("pitch"), item.get("rhythm"))
                for item in attempt.get("context", [])
            ]
            context_label = " → ".join(format_event(event) for event in context_events) if context_events else "None"

            if status == "used":
                status_text = "used"
            elif status == "too_rare":
                status_text = f"too rare ({total_count}/{min_count})"
            elif status == "missing":
                status_text = "missing"
            else:
                status_text = status

            attempt_lines.append(f"{order}-event memory: {context_label} — {status_text}")

        selected_event_from_trace = trace_item.get("selected_event", {})
        selected_from_trace = (
            selected_event_from_trace.get("pitch", selected_event[0]),
            selected_event_from_trace.get("rhythm", selected_event[1]),
        )

        possible_next_events = trace_item.get("possible_next_events", [])
        choices = [
            (
                (item.get("pitch"), item.get("rhythm")),
                float(item.get("probability", 0.0)),
            )
            for item in possible_next_events
        ]

        used_order = trace_item.get("used_order", 0)
        if used_order:
            memory = "<br>".join(attempt_lines)
            explanation = trace_item.get(
                "explanation",
                f"The model used the longest reliable memory it could find: order {used_order}.",
            )
        else:
            memory = "<br>".join(attempt_lines) if attempt_lines else "No reliable Markov memory was found."
            explanation = trace_item.get(
                "explanation",
                "No reliable Markov pattern was found, so the model used weighted-random fallback.",
            )

        return {
            "memory": memory,
            "choices": normalize_choices(choices),
            "selected": selected_from_trace,
            "selected_label": format_event(selected_from_trace),
            "explanation": explanation,
            "fallback": trace_item.get("fallback_used", False),
            "choice_kind": "event_probability",
        }

    if model_name == "RuleBased":
        if step == 0:
            pitch_classes = getattr(model_instance, "pitch_classes", [])
            style_name = getattr(getattr(model_instance, "settings", {}), "get", lambda k, d=None: d)("display_name", "selected style")
            return {
                "memory": f"Starting note. Style: {style_name}. Pitch pool: {', '.join(pitch_classes)}.",
                "choices": [],
                "selected": selected_pitch,
                "selected_label": selected_pitch,
                "explanation": f"The rule-based melody starts on {selected_pitch}, the starting note for the selected style.",
                "fallback": False,
                "choice_kind": "rule",
            }

        trace_item = model_instance.get_trace_for_step(step) if hasattr(model_instance, "get_trace_for_step") else None

        if not trace_item:
            return {
                "memory": "Rule trace was not stored for this step.",
                "choices": [],
                "selected": selected_pitch,
                "selected_label": selected_pitch,
                "explanation": "This rule-based step is not available in the saved trace.",
                "fallback": False,
                "choice_kind": "rule",
            }

        candidates = trace_item.get("possible_next_notes", [])
        if candidates:
            raw_scores = [float(item.get("score", 0.0)) for item in candidates]
            min_score = min(raw_scores)
            max_score = max(raw_scores)
            span = max(max_score - min_score, 1.0)
            choices = [
                (item.get("note"), (float(item.get("score", 0.0)) - min_score) / span, float(item.get("score", 0.0)))
                for item in candidates
            ]
        else:
            choices = []

        melody_so_far = trace_item.get("melody_so_far", [])
        pitch_classes = trace_item.get("pitch_classes", getattr(model_instance, "pitch_classes", []))
        style_name = trace_item.get("style_name", "selected style")

        return {
            "memory": (
                f"Melody so far: {' → '.join(melody_so_far) if melody_so_far else 'None yet'}<br>"
                f"Style: {style_name}<br>"
                f"Pitch pool: {', '.join(pitch_classes)}"
            ),
            "choices": choices,
            "selected": selected_pitch,
            "selected_label": selected_pitch,
            "explanation": trace_item.get("explanation", f"{selected_pitch} had the best rule-based score for this step."),
            "fallback": False,
            "choice_kind": "rule",
        }

    return {
        "memory": "No model explanation available.",
        "choices": [],
        "selected": selected_event,
        "selected_label": format_event(selected_event),
        "explanation": "This model is not configured for Learning Mode yet.",
        "fallback": False,
    }


def render_melody_builder(generated_events, step):
    """Render melody so far as note-event chips."""
    chips = []
    for i, event in enumerate(generated_events):
        if i < step:
            css_class = "note-chip"
            label = format_event(event)
        elif i == step:
            css_class = "note-chip current"
            label = format_event(event)
        else:
            css_class = "note-chip future"
            label = "_"
        chips.append(f'<div class="{css_class}">{label}</div>')

    st.markdown(
        f"""
        <div class="melody-builder">
            {''.join(chips)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_choice_bars(choices, selected_item, choice_kind="probability"):
    """Render possible next note-events as probability bars or notes as relative rule-score bars."""
    if not choices:
        message = "No probability table is needed for this starting step."
        if choice_kind == "rule":
            message = "No rule-score table is needed for the starting note."
        st.markdown(
            f"""
            <div class="learning-card-body">
                {message}
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    rows = []
    for choice in choices:
        if len(choice) == 3:
            item, value, raw_score = choice
        else:
            item, value = choice
            raw_score = None

        percent = int(round(max(0.0, min(1.0, float(value))) * 100))
        selected_marker = " ✅" if item == selected_item else ""

        if choice_kind == "rule" and raw_score is not None:
            item_label = str(item)
            value_label = f"score {raw_score:g}"
        else:
            item_label = format_event(item)
            value_label = f"{percent}%"

        rows.append(
            f"""
            <div class="choice-row">
                <div class="choice-label">
                    <span>{item_label}{selected_marker}</span>
                    <span>{value_label}</span>
                </div>
                <div class="choice-track">
                    <div class="choice-fill" style="width:{percent}%;"></div>
                </div>
            </div>
            """
        )

    st.markdown("".join(rows), unsafe_allow_html=True)


def page_learning():
    """Interactive note-by-note replay of the generated melody."""
    inject_global_ui_css()
    render_page_header("Watch how it was composed", "Step through the generated melody and inspect the model’s memory, choices, and selected note.")

    melody = st.session_state.generated_melody
    model_instance = st.session_state.model_instance

    if melody is None or model_instance is None:
        st.warning("Generate a melody first.")
        if st.button("Back to Generate", use_container_width=True):
            st.session_state.page = "generate"
            st.rerun()
        return

    inject_learning_css()

    generated_events = [(pitch, duration) for pitch, duration in melody]
    generated_pitches = [pitch for pitch, duration in melody]
    generated_durations = [duration for pitch, duration in melody]
    total_notes = len(generated_events)

    step = int(st.session_state.learning_step)
    step = max(0, min(step, total_notes - 1))
    st.session_state.learning_step = step

    selected_pitch = generated_pitches[step]
    selected_duration = generated_durations[step]

    if st.session_state.play_learning_note:
        autoplay_single_note(selected_pitch, selected_duration, tempo=120)
        st.session_state.play_learning_note = False

    info = get_learning_step_info(
        st.session_state.selected_model,
        model_instance,
        generated_events,
        generated_pitches,
        step,
    )

    st.markdown(
        f"""
        <div class="learning-hero">
            <div class="learning-title">🎼 Melody Builder</div>
            <div>
                <b>{get_model_friendly_name(st.session_state.selected_model)}</b><br>
                Step {step + 1} of {total_notes}: revealing the selected note-event and how the model chose it.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_melody_builder(generated_events, step)

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown(
            f"""
            <div class="learning-card">
                <div class="learning-card-title">🧠 Model memory</div>
                <div class="learning-card-body">{info["memory"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="learning-card">
                <div class="learning-card-title">✅ Selected note</div>
                <div class="learning-card-body">
                    <span class="selected-note">{info["selected_label"]}</span><br><br>
                    Pitch: <b>{selected_pitch}</b><br>
                    Duration: <b>{selected_duration}</b> beat(s)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            """
            <div class="learning-card">
                <div class="learning-card-title">🎲 Possible next notes</div>
            """,
            unsafe_allow_html=True,
        )
        selected_choice = info["selected"]
        render_choice_bars(info["choices"], selected_choice, info.get("choice_kind", "probability"))
        st.markdown("</div>", unsafe_allow_html=True)

        fallback_note = ""
        if info.get("fallback"):
            fallback_note = "<br><br><b>Note:</b> This was a fallback/backoff step because the exact memory pattern had no stored continuation."

        st.markdown(
            f"""
            <div class="learning-card">
                <div class="learning-card-title">💡 Why?</div>
                <div class="learning-card-body">{info["explanation"]}{fallback_note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    progress_value = (step + 1) / total_notes if total_notes else 0
    st.progress(progress_value, text=f"Composition replay progress: {step + 1}/{total_notes} notes")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("← Back to Results", use_container_width=True):
            st.session_state.play_learning_note = False
            st.session_state.page = "results"
            st.rerun()

    with col2:
        if st.button("Restart Replay", use_container_width=True):
            st.session_state.learning_step = 0
            st.session_state.play_learning_note = False
            st.rerun()

    with col3:
        if step < total_notes - 1:
            if st.button("Think Next Note →", use_container_width=True):
                st.session_state.learning_step += 1
                st.session_state.play_learning_note = True
                st.rerun()
        else:
            st.success("Full melody composed!")
            if st.button("Play Full Melody Again", use_container_width=True):
                audio_bytes = melody_to_wav(melody, tempo=120)
                render_play_melody_button(audio_bytes)



# Page: RESULTS - Scorecard Display
def page_results():
    inject_global_ui_css()

    if st.button("← Back to model selection", key="results_back_home"):
        st.session_state.page = "home"
        st.session_state.selected_model = None
        st.session_state.selected_melody = None
        st.session_state.model_instance = None
        st.session_state.generated_melody = None
        st.session_state.scorecard_results = None
        st.session_state.rule_mode = None
        st.rerun()

    render_page_header("Melody analysis", "Listen to the result, inspect the notes, and compare the music-theory scorecard.")
    
    melody = st.session_state.generated_melody
    results = st.session_state.scorecard_results

    if melody is None or results is None:
        st.warning("No melody analysis is available yet. Generate a melody first.")
        if st.button("Back to Generate", use_container_width=True):
            st.session_state.page = "generate"
            st.rerun()
        return
    
    # Display melody sequence and click-to-play audio
    st.write("### Generated Melody")
    st.markdown(
    """
    <div style="
        margin-top: 0.35rem;
        margin-bottom: 0.9rem;
        padding: 0.65rem 0.85rem;
        border-left: 4px solid #c4b5fd;
        border-radius: 10px;
        background: rgba(248, 250, 252, 0.9);
        color: #475569;
        font-size: 0.94rem;
        line-height: 1.45;
    ">
        This is the melody produced by the model. The first line shows the pitch in music notation, and the small number below it shows the beat duration.
    </div>
    """,
    unsafe_allow_html=True,
)

    audio_bytes = melody_to_wav(melody, tempo=120)
    render_play_melody_button(audio_bytes)

    cols = st.columns(8)
    for i, (pitch, duration) in enumerate(melody):
        with cols[i % 8]:
            st.write(f"**{pitch}**")
            st.caption(f"{duration}♪")
    
    st.divider()
    
    # Display scorecard
    st.write("### Melody Feel Scorecard")
    
    smoothness = results["interval_smoothness"]
    jumps = 1.0 - smoothness  # Inverse
    patterns = results["motif_repetition"]
    rhythm = results["rhythmic_variety"]
    overall = results["final_score"]
    
    # Polished visual scorecard display
    inject_scorecard_css()

    overall_percent = int(round(clamp_score(overall) * 100))
    overall_category = get_category(overall, "standard")
    st.markdown(
        f"""
        <div class="score-hero">
            <div class="score-hero-title">✨ Overall feel</div>
            <div class="score-hero-number">{overall_percent}/100</div>
            <div class="score-hero-subtitle">
                Based on whether the melody moves smoothly, avoids too many jumps, repeats ideas, and uses varied beats.
                Current level: <b>{overall_category}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Note: this scorecard is a guide, not a judge. A high score means the melody matches patterns MelodyLab can measure, "
        "but your ear still matters. Some melodies can follow the rules and still sound plain, while others may break rules and sound interesting."
    )

    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        metric_card(
            "Does it move smoothly?",
            smoothness,
            get_category(smoothness, "standard"),
            "Higher means the melody mostly moves between nearby pitches instead of sudden jumps.",
            "🌊",
        )
    with row1_col2:
        metric_card(
            "Does it jump around?",
            jumps,
            get_category(jumps, "standard"),
            "Higher means the melody uses more wide pitch changes. A few can be exciting; too many can feel random.",
            "🦘",
        )
    with row1_col3:
        metric_card(
            "Does it repeat ideas?",
            patterns,
            get_category(patterns, "standard"),
            "Higher means the melody brings back short patterns, which can make it feel more organized.",
            "🔁",
        )

    rhythm_col, _spacer_col = st.columns([1, 2])
    with rhythm_col:
        metric_card(
            "Are the beats varied?",
            rhythm,
            get_category(rhythm, "rhythm"),
            "Checks whether the beat durations have some variety without becoming completely scattered.",
            "🥁",
        )

    with st.expander("Explanation for nerds", expanded=False):
        st.markdown("""
            Smoothness rewards mostly stepwise motion. Jumps is the inverse of smoothness, so a high jump score means the melody contains more large leaps. Patterns measures motif repetition. Beat Variety combines rhythmic diversity with rhythmic patterning.
            """
        )

    st.divider()
    
    # Main feature buttons
    primary_col1, primary_col2 = st.columns(2)

    with primary_col1:
        if st.button("🎼 Watch How It Was Composed", use_container_width=True, type="primary"):
            st.session_state.learning_step = 0
            st.session_state.play_learning_note = False
            st.session_state.page = "learning"
            st.rerun()

    with primary_col2:
        if st.button(
            "⚖️ Compare Melodies",
            use_container_width=True,
            type="primary",
            disabled=len(st.session_state.melody_history) < 2,
        ):
            st.session_state.page = "compare"
            st.rerun()

    # Secondary actions
    secondary_col1, secondary_col2 = st.columns(2)

    with secondary_col1:
        if st.button("Generate Again", use_container_width=True):
            st.session_state.page = "generate"
            st.session_state.generated_melody = None
            st.session_state.scorecard_results = None
            st.rerun()

    with secondary_col2:
        if st.button("📜 View Melody History", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()

# Router
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "rule_mode":
    page_rule_mode()
elif st.session_state.page == "training_melody":
    page_training_melody()
elif st.session_state.page == "generate":
    page_generate()
elif st.session_state.page == "results":
    page_results()
elif st.session_state.page == "learning":
    page_learning()
elif st.session_state.page == "history":
    page_history()
elif st.session_state.page == "compare":
    page_compare()
