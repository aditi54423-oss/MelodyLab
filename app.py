import streamlit as st
from utils.melodies import MINUET_PITCHES, MINUET_RHYTHM
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
        st.success("✅ Melody generated and stored!")
        st.info(f"Generated {len(melody)} notes")
        
        # Display the generated melody
        st.write("### Generated Melody (Stored)")
        for i, (pitch, duration) in enumerate(melody, 1):
            st.write(f"Note {i}: {pitch} ({duration} beats)")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Try Another Model", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.selected_model = None
            st.session_state.selected_melody = None
            st.session_state.model_instance = None
            st.session_state.generated_melody = None
            st.rerun()
    
    with col2:
        if st.button("Generate Again →", use_container_width=True):
            # This will trigger another generation on next rerun
            st.rerun()

# Router
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "training_melody":
    page_training_melody()
elif st.session_state.page == "generate":
    page_generate()
