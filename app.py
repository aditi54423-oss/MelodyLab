import streamlit as st
from utils.melodies import MINUET_PITCHES, MINUET_RHYTHM
from utils.scorecard import evaluate_melody, infer_duration_vocab_size
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
        d_max = infer_duration_vocab_size([MINUET_PITCHES], [MINUET_RHYTHM])
        scorecard_results = evaluate_melody(melody, d_max)
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

# Page: RESULTS - Scorecard Display
def page_results():
    st.title("🎵 MelodyLab")
    st.subheader("Melody Analysis")
    
    melody = st.session_state.generated_melody
    results = st.session_state.scorecard_results
    
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
    
    # Create scorecard display
    scorecard_lines = []
    scorecard_lines.append("┌─────────────────────────────────────┐")
    scorecard_lines.append("│ Melody Scorecard                    │")
    scorecard_lines.append("│                                     │")
    
    # Smoothness
    bar = create_bar(smoothness)
    category = get_category(smoothness, "standard")
    scorecard_lines.append(f"│ Smoothness      {bar} {category:>8} │")
    
    # Jumps
    bar = create_bar(jumps)
    category = get_category(jumps, "standard")
    scorecard_lines.append(f"│ Jumps           {bar} {category:>8} │")
    
    # Patterns
    bar = create_bar(patterns)
    category = get_category(patterns, "standard")
    scorecard_lines.append(f"│ Patterns        {bar} {category:>8} │")
    
    # Ending
    bar = create_bar(ending)
    category = get_category(ending, "ending")
    scorecard_lines.append(f"│ Ending                  {category:>15} │")
    
    # Beat Variety
    bar = create_bar(rhythm)
    category = get_category(rhythm, "rhythm")
    scorecard_lines.append(f"│ Beat Variety    {bar} {category:>8} │")
    
    scorecard_lines.append("│                                     │")
    
    # Overall Score
    overall_percent = int(overall * 100)
    scorecard_lines.append(f"│ Overall Score           {overall_percent:>3}/100        │")
    scorecard_lines.append("│                                     │")
    scorecard_lines.append("└─────────────────────────────────────┘")
    
    # Display scorecard in monospace
    scorecard_text = "\n".join(scorecard_lines)
    st.code(scorecard_text, language="")
    
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
