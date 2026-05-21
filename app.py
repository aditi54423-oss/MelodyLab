import streamlit as st
from utils.melodies import MINUET_PITCHES, MINUET_RHYTHM
from utils.scorecard import evaluate_sequence, infer_duration_vocab_size
from models.random_model import RandomMelodyGenerator
from models.markov1 import FirstOrderMarkovGenerator
from models.markov2 import SecondOrderMarkovGenerator

st.set_page_config(page_title="MelodyLab", layout="wide", initial_sidebar_state="collapsed")

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

# Page: HOME - Model Selection
def page_home():
    st.title("🎵 MelodyLab")
    st.subheader("Can Math Make Music?")
    st.write("Try generating a melody with different models.")
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎲 Random\n*The Dice Roller*\nChooses notes by chance", use_container_width=True, key="btn_random"):
            st.session_state.selected_model = "Random"
            st.session_state.page = "training_melody"
            st.rerun()
    
    with col2:
        if st.button("🎵 First-Order Markov\n*The One-Note Listener*\nRemembers the previous note", use_container_width=True, key="btn_markov1"):
            st.session_state.selected_model = "Markov1"
            st.session_state.page = "training_melody"
            st.rerun()
    
    with col3:
        if st.button("🎼 Second-Order Markov\n*The Pattern Imitator*\nRemembers the previous two notes", use_container_width=True, key="btn_markov2"):
            st.session_state.selected_model = "Markov2"
            st.session_state.page = "training_melody"
            st.rerun()

# Page: TRAINING MELODY Selection
def page_training_melody():
    st.title("🎵 MelodyLab")
    st.subheader(f"Choose Training Melody")
    st.write(f"Selected Model: **{st.session_state.selected_model}**")
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.write("### Available Training Melodies")
        
        melody_options = {
            "Minuet (Bach)": "minuet"
        }
        
        for melody_name, melody_key in melody_options.items():
            if st.button(melody_name, use_container_width=True, key=f"btn_{melody_key}"):
                st.session_state.selected_melody = melody_key
                st.session_state.page = "generate"
                st.rerun()
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.selected_model = None
            st.rerun()

# Page: GENERATE Melody
def page_generate():
    st.title("🎵 MelodyLab")
    st.subheader("Generate Melody")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", st.session_state.selected_model)
    with col2:
        melody_display = st.session_state.selected_melody.replace("_", " ").title()
        st.metric("Training Melody", melody_display)
    with col3:
        st.metric("Length", "16 notes")
    
    st.divider()
    
    # Initialize model instance if needed
    if st.session_state.model_instance is None:
        if st.session_state.selected_model == "Random":
            st.session_state.model_instance = RandomMelodyGenerator(
                pitches=MINUET_PITCHES,
                rhythms=MINUET_RHYTHM
            )
        elif st.session_state.selected_model == "Markov1":
            st.session_state.model_instance = FirstOrderMarkovGenerator(
                pitches=MINUET_PITCHES,
                rhythms=MINUET_RHYTHM
            )
        elif st.session_state.selected_model == "Markov2":
            st.session_state.model_instance = SecondOrderMarkovGenerator(
                pitches=MINUET_PITCHES,
                rhythms=MINUET_RHYTHM
            )
    
    # Generate Melody Button
    if st.button("🎵 Generate Melody", use_container_width=True, key="btn_generate"):
        melody = st.session_state.model_instance.generate_melody(length=16)
        st.session_state.generated_melody = melody
        
        # Evaluate melody with scorecard
        training_melody = list(zip(MINUET_PITCHES, MINUET_RHYTHM))
        d_max = infer_duration_vocab_size([training_melody])
        scorecard_results = evaluate_sequence(melody, d_max)
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
        return "Finished on Tonic" if score >= 0.5 else "Unresolved Ending"
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

# Page: RESULTS - Scorecard Display
def page_results():
    st.title("🎵 MelodyLab")
    st.subheader("Melody Analysis")
    
    melody = st.session_state.generated_melody
    results = st.session_state.scorecard_results

    if melody is None or results is None:
        st.warning("No melody analysis is available yet. Generate a melody first.")
        if st.button("Back to Generate", use_container_width=True):
            st.session_state.page = "generate"
            st.rerun()
        return
    
    # Display melody sequence
    st.write("### Generated Melody")
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
                Based on smoothness, melodic jumps, recurring patterns, ending resolution, and rhythmic variety.
                Current level: <b>{overall_category}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
            "Whether the melody resolves on the inferred tonic.",
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

    with st.expander("What do these scores mean?"):
        st.write(
            "Smoothness rewards mostly stepwise motion. Jumps is the inverse of smoothness, "
            "so a high jump score means the melody contains more large leaps. Patterns measures "
            "motif repetition. Ending checks whether the final note resolves on the inferred tonic. "
            "Beat Variety combines rhythmic diversity with rhythmic patterning."
        )

    st.divider()
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎵 Generate Again", use_container_width=True):
            st.session_state.page = "generate"
            st.session_state.generated_melody = None
            st.session_state.scorecard_results = None
            st.rerun()
    
    with col2:
        if st.button("Watch How It Was Composed", use_container_width=True):
            st.info("Coming soon! Step-by-step breakdown of how the melody was generated.")
    
    with col3:
        if st.button("← Try Another Model", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.selected_model = None
            st.session_state.selected_melody = None
            st.session_state.model_instance = None
            st.session_state.generated_melody = None
            st.session_state.scorecard_results = None
            st.rerun()

# Router
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "training_melody":
    page_training_melody()
elif st.session_state.page == "generate":
    page_generate()
elif st.session_state.page == "results":
    page_results()
