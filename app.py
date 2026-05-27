import streamlit as st
import numpy as np
import io
import wave
import base64
from pathlib import Path
import streamlit.components.v1 as components
from utils.melodies import TRAINING_MELODIES
from utils.scorecard import evaluate_sequence, infer_duration_vocab_size
from models.random_model import RandomMelodyGenerator
from models.markov1 import FirstOrderMarkovGenerator
from models.markov2 import SecondOrderMarkovGenerator
from models.rule_based_basic import RuleBasedMelodyGenerator

APP_DIR = Path(__file__).resolve().parent
LOGO_FILE = APP_DIR / "melodylab_logo.png"
FAVICON_FILE = APP_DIR / "melodylab_favicon.png"

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

# Home-note settings for the scorecard.
# These are used instead of automatic Western key detection, so Sakura Sakura
# can be judged by its own resting/home note.
DEFAULT_HOME_NOTES = {
    "minuet": "G",
    "amazing_grace": "F",
    "amazing grace": "F",
    "sakura": "A",
    "sakura_sakura": "A",
    "sakura sakura": "A",
}


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


def get_training_home_note(melody_key, melody_data):
    """
    Return the expected home note for the selected training melody.
    Priority:
    1. A home_note field inside TRAINING_MELODIES, if you add one later.
    2. A safe name/key lookup for the current built-in melodies.
    3. G as a fallback for older Minuet-only versions.
    """
    if "home_note" in melody_data:
        return str(melody_data["home_note"]).strip()

    key_text = str(melody_key).lower().strip()
    name_text = str(melody_data.get("name", "")).lower().strip()

    for label, home_note in DEFAULT_HOME_NOTES.items():
        if label in key_text or label in name_text:
            return home_note

    return "G"


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
        "meaning": "Chooses notes by chance from the selected melody’s note pool.",
        "button": "Choose Random",
    },
    "Markov1": {
        "name": "First-Order Markov",
        "personality": "The One-Note Listener",
        "meaning": "Looks at the previous note before choosing what comes next.",
        "button": "Choose First-Order Markov",
    },
    "Markov2": {
        "name": "Second-Order Markov",
        "personality": "The Pattern Imitator",
        "meaning": "Looks at the previous two notes, so it can imitate short patterns.",
        "button": "Choose Second-Order Markov",
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

# Page: HOME - Model Selection
def page_home():
    inject_global_ui_css()
    render_brand_hero()
    render_page_header("Choose a model", "Each model has a different composing personality. Start with one and compare the results later.")

    cols = st.columns(4)
    model_keys = ["Random", "RuleBased", "Markov1", "Markov2"]
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
    render_page_header("Choose a training melody", "The selected melody becomes the source material for the model’s note and rhythm choices.")
    st.markdown(
        f'Selected Model: <span class="ml-selected-pill">{get_model_display_name(st.session_state.selected_model)}</span>',
        unsafe_allow_html=True,
    )
    if st.session_state.selected_model == "RuleBased":
        st.markdown(
            f'Rule-Based Mode: <span class="ml-selected-pill">{(st.session_state.rule_mode or "strict").title()} Mode</span>',
            unsafe_allow_html=True,
        )
        st.caption("For this model, the training melody sets the pitch-class pool and home note.")
    st.divider()

    melody_options = {data["name"]: key for key, data in TRAINING_MELODIES.items()}
    cols = st.columns(3)

    for idx, (melody_name, melody_key) in enumerate(melody_options.items()):
        melody_data = TRAINING_MELODIES[melody_key]
        note_count = len(melody_data.get("pitches", []))
        melody_icon = get_training_melody_icon(melody_key, melody_data)

        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="ml-card">
                    <div class="ml-card-title">{melody_icon} {melody_name}</div>
                    <div class="ml-card-text">{note_count} training notes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Train on {melody_icon} {melody_name}", use_container_width=True, key=f"btn_{melody_key}"):
                st.session_state.selected_melody = melody_key
                st.session_state.model_instance = None
                st.session_state.page = "generate"
                st.rerun()

    st.divider()
    if st.button("← Back to model selection", use_container_width=True):
        st.session_state.page = "home"
        st.session_state.selected_model = None
        st.rerun()

# Page: GENERATE Melody
def page_generate():
    inject_global_ui_css()
    render_page_header("Generate melody", "Review your setup, then create a melody from the selected model.")

    melody_display = TRAINING_MELODIES[st.session_state.selected_melody]["name"]

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
        selected_melody_data = TRAINING_MELODIES[st.session_state.selected_melody]
        pitches = selected_melody_data["pitches"]
        rhythms = selected_melody_data["rhythms"]
        
        if st.session_state.selected_model == "Random":
            st.session_state.model_instance = RandomMelodyGenerator(
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
        elif st.session_state.selected_model == "RuleBased":
            st.session_state.model_instance = RuleBasedMelodyGenerator(
                training_melody_key=st.session_state.selected_melody,
                mode=st.session_state.rule_mode or "strict"
            )
    
    # Generate Melody Button
    if st.button("✨ Generate Melody →", use_container_width=True, key="btn_generate", type="primary"):
        melody = st.session_state.model_instance.generate_melody(length=st.session_state.melody_length)
        st.session_state.generated_melody = melody
        st.session_state.learning_step = 0
        st.session_state.play_learning_note = False
        
        # Evaluate melody with scorecard
        selected_melody_data = TRAINING_MELODIES[st.session_state.selected_melody]
        training_melody = list(zip(selected_melody_data["pitches"], selected_melody_data["rhythms"]))
        d_max = infer_duration_vocab_size([training_melody])
        home_note = get_training_home_note(st.session_state.selected_melody, selected_melody_data)
        scorecard_results = evaluate_sequence(melody, d_max, home_note=home_note)
        st.session_state.scorecard_results = scorecard_results
        
        # Move to results page
        st.session_state.page = "results"
        st.rerun()
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Try Another Model", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.selected_model = None
            st.session_state.selected_melody = None
            st.session_state.model_instance = None
            st.session_state.generated_melody = None
            st.session_state.scorecard_results = None
            st.session_state.learning_step = 0
            st.session_state.rule_mode = None
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
    elif metric_type == "ending":
        return "Finished on Home Note" if score >= 0.5 else "Unresolved Ending"
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

    if len(note) >= 3 and note[1] in ["#", "b"]:
        pitch_class = note[:2].upper()
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
    components.html(
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

    components.html(
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


def normalize_choices(choices):
    """Sort and clean a list of (choice, probability) pairs."""
    cleaned = []
    for note, prob in choices:
        try:
            p = float(prob)
        except (TypeError, ValueError):
            p = 0.0
        cleaned.append((note, max(0.0, min(1.0, p))))
    return sorted(cleaned, key=lambda item: item[1], reverse=True)


def get_learning_step_info(model_name, model_instance, generated_pitches, step):
    """
    Explain how the current pitch could have been selected.

    This reads the already-built chains from the model instance, so it stays aligned
    with the generator classes without changing those model files.
    """
    selected_note = generated_pitches[step]

    if model_name == "Random":
        pitch_pool = getattr(model_instance, "pitch_pool", sorted(set(getattr(model_instance, "pitches", generated_pitches))))
        probability = 1.0 / len(pitch_pool) if pitch_pool else 0.0
        return {
            "memory": "No memory. This model chooses each pitch independently.",
            "choices": normalize_choices([(note, probability) for note in pitch_pool]),
            "selected": selected_note,
            "explanation": f"{selected_note} was selected by chance from the available pitch pool.",
            "fallback": False,
        }

    if model_name == "Markov1":
        if step == 0:
            return {
                "memory": "No previous note yet.",
                "choices": [],
                "selected": selected_note,
                "explanation": f"The melody starts with {selected_note}. The first note is used as the starting point before the one-note memory begins.",
                "fallback": False,
            }

        previous_note = generated_pitches[step - 1]
        chain = getattr(model_instance, "note_chain", {})
        choices = chain.get(previous_note, [])

        if choices:
            explanation = f"The model looked at the previous note, {previous_note}, and used the training melody's transition probabilities."
            fallback = False
        else:
            keys = sorted(chain.keys())
            probability = 1.0 / len(keys) if keys else 0.0
            choices = [(note, probability) for note in keys]
            explanation = f"The model had no stored next-note options after {previous_note}, so it restarted from a valid training-state note."
            fallback = True

        return {
            "memory": f"Previous note: {previous_note}",
            "choices": normalize_choices(choices),
            "selected": selected_note,
            "explanation": explanation,
            "fallback": fallback,
        }

    if model_name == "Markov2":
        if step == 0:
            return {
                "memory": "No two-note memory yet.",
                "choices": [],
                "selected": selected_note,
                "explanation": f"The melody starts with {selected_note}. The second-order model first needs a two-note starting pair.",
                "fallback": False,
            }

        if step == 1:
            previous_note = generated_pitches[step - 1]
            return {
                "memory": f"Starting pair being formed: {previous_note} → {selected_note}",
                "choices": [],
                "selected": selected_note,
                "explanation": f"{selected_note} completes the starting pair. After this, the model can use two-note memory.",
                "fallback": False,
            }

        previous_pair = (generated_pitches[step - 2], generated_pitches[step - 1])
        chain = getattr(model_instance, "note_chain", {})
        choices = chain.get(previous_pair, [])

        if choices:
            explanation = (
                f"The model looked at the two-note pattern {previous_pair[0]} → {previous_pair[1]} "
                "and used the training melody's transition probabilities."
            )
            fallback = False
        else:
            keys = sorted(chain.keys())
            probability = 1.0 / len(keys) if keys else 0.0
            choices = [(pair[1], probability) for pair in keys]
            explanation = (
                f"The model had not seen {previous_pair[0]} → {previous_pair[1]} as a stored pattern, "
                "so it restarted from a valid training-state pair."
            )
            fallback = True

        return {
            "memory": f"Previous two notes: {previous_pair[0]} → {previous_pair[1]}",
            "choices": normalize_choices(choices),
            "selected": selected_note,
            "explanation": explanation,
            "fallback": fallback,
        }

    if model_name == "RuleBased":
        if step == 0:
            pitch_classes = getattr(model_instance, "pitch_classes", [])
            style_name = getattr(getattr(model_instance, "settings", {}), "get", lambda k, d=None: d)("display_name", "selected style")
            return {
                "memory": f"Starting home note. Style: {style_name}. Pitch pool: {', '.join(pitch_classes)}.",
                "choices": [],
                "selected": selected_note,
                "explanation": f"The rule-based melody starts on {selected_note}, the home note for the selected style.",
                "fallback": False,
                "choice_kind": "rule",
            }

        trace_item = model_instance.get_trace_for_step(step) if hasattr(model_instance, "get_trace_for_step") else None

        if not trace_item:
            return {
                "memory": "Rule trace was not stored for this step.",
                "choices": [],
                "selected": selected_note,
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
            "selected": selected_note,
            "explanation": trace_item.get("explanation", f"{selected_note} had the best rule-based score for this step."),
            "fallback": False,
            "choice_kind": "rule",
        }

    return {
        "memory": "No model explanation available.",
        "choices": [],
        "selected": selected_note,
        "explanation": "This model is not configured for Learning Mode yet.",
        "fallback": False,
    }


def render_melody_builder(generated_pitches, step):
    """Render melody so far as note chips."""
    chips = []
    for i, note in enumerate(generated_pitches):
        if i < step:
            css_class = "note-chip"
            label = note
        elif i == step:
            css_class = "note-chip current"
            label = note
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


def render_choice_bars(choices, selected_note, choice_kind="probability"):
    """Render possible next notes as probability bars or relative rule-score bars."""
    if not choices:
        message = "No probability table is needed for this starting step."
        if choice_kind == "rule":
            message = "No rule-score table is needed for the starting home note."
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
            note, value, raw_score = choice
        else:
            note, value = choice
            raw_score = None

        percent = int(round(max(0.0, min(1.0, float(value))) * 100))
        selected_marker = " ✅" if note == selected_note else ""

        if choice_kind == "rule" and raw_score is not None:
            value_label = f"score {raw_score:g}"
        else:
            value_label = f"{percent}%"

        rows.append(
            f"""
            <div class="choice-row">
                <div class="choice-label">
                    <span>{note}{selected_marker}</span>
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

    generated_pitches = [pitch for pitch, duration in melody]
    generated_durations = [duration for pitch, duration in melody]
    total_notes = len(generated_pitches)

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
        generated_pitches,
        step,
    )

    st.markdown(
        f"""
        <div class="learning-hero">
            <div class="learning-title">🎼 Melody Builder</div>
            <div>
                <b>{get_model_friendly_name(st.session_state.selected_model)}</b><br>
                Step {step + 1} of {total_notes}: revealing the selected pitch and how the model chose it.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_melody_builder(generated_pitches, step)

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
                    <span class="selected-note">{info["selected"]}</span><br><br>
                    Duration used in the generated melody: <b>{selected_duration}</b> beat(s)
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
        render_choice_bars(info["choices"], selected_pitch, info.get("choice_kind", "probability"))
        st.markdown("</div>", unsafe_allow_html=True)

        fallback_note = ""
        if info.get("fallback"):
            fallback_note = "<br><br><b>Note:</b> This was a fallback/restart step because the exact memory pattern had no stored continuation."

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

    audio_bytes = melody_to_wav(melody, tempo=120)
    render_play_melody_button(audio_bytes)

    cols = st.columns(8)
    for i, (pitch, duration) in enumerate(melody):
        with cols[i % 8]:
            st.write(f"**{pitch}**")
            st.caption(f"{duration}♪")
    
    st.divider()
    
    # Display scorecard
    st.write("### Music Theory Scorecard")
    
    smoothness = results["interval_smoothness"]
    jumps = 1.0 - smoothness  # Inverse
    patterns = results["motif_repetition"]
    ending = results["ending_on_tonic"]
    rhythm = results["rhythmic_variety"]
    overall = results["final_score"]
    
    # Polished visual scorecard display
    inject_scorecard_css()

    overall_percent = int(round(clamp_score(overall) * 100))
    overall_category = get_category(overall, "standard")
    st.markdown(
        f"""
        <div class="score-hero">
            <div class="score-hero-title">✨ Overall musicality</div>
            <div class="score-hero-number">{overall_percent}/100</div>
            <div class="score-hero-subtitle">
                Based on smoothness, melodic jumps, recurring patterns, home-note resolution, and rhythmic variety.
                Current level: <b>{overall_category}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
    "Note: a high score means the melody matches several features the scorecard checks, "
    "such as smooth movement, recurring patterns, rhythmic variety, and ending on the home note. "
    "It does not always mean the melody will sound the most musical to a listener. "
    "A melody can follow the rules well but still feel repetitive, which shows why listening matters alongside the math."
)

    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        metric_card(
            "Smoothness",
            smoothness,
            get_category(smoothness, "standard"),
            "How connected the note-to-note movement feels.",
            "🌊",
        )
    with row1_col2:
        metric_card(
            "Jumps",
            jumps,
            get_category(jumps, "standard"),
            "How much the melody uses larger leaps.",
            "🦘",
        )
    with row1_col3:
        metric_card(
            "Patterns",
            patterns,
            get_category(patterns, "standard"),
            "How strongly motifs or repeated ideas appear.",
            "🔁",
        )

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        metric_card(
            "Ending",
            ending,
            get_category(ending, "ending"),
            "Whether the melody resolves on the selected training melody’s home note.",
            "🏁",
        )
    with row2_col2:
        metric_card(
            "Beat Variety",
            rhythm,
            get_category(rhythm, "rhythm"),
            "Balance between rhythmic variety and repeated beat patterns.",
            "🥁",
        )

    with st.expander("What do these scores mean?", expanded=False):
        st.write(
            "Smoothness rewards mostly stepwise motion. Jumps is the inverse of smoothness, "
            "so a high jump score means the melody contains more large leaps. Patterns measures "
            "motif repetition. Ending checks whether the final note resolves on the selected training melody’s home note. "
            "Beat Variety combines rhythmic diversity with rhythmic patterning."
        )

    st.divider()
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Generate Again", use_container_width=True):
            st.session_state.page = "generate"
            st.session_state.generated_melody = None
            st.session_state.scorecard_results = None
            st.rerun()
    
    with col2:
        if st.button("🎼 Watch How It Was Composed", use_container_width=True, type="primary"):
            st.session_state.learning_step = 0
            st.session_state.play_learning_note = False
            st.session_state.page = "learning"
            st.rerun()
    
    with col3:
        if st.button("← Try Another Model", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.selected_model = None
            st.session_state.selected_melody = None
            st.session_state.model_instance = None
            st.session_state.generated_melody = None
            st.session_state.scorecard_results = None
            st.session_state.rule_mode = None
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
