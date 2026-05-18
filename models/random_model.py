import random

class RandomMelodyGenerator:
    """Random melody generator - chooses notes by chance."""
    
    def __init__(self, pitches, rhythms):
        """
        Initialize Random melody generator.
        
        Args:
            pitches: List of available pitches
            rhythms: List of available rhythms (durations)
        """
        self.pitches = pitches
        self.rhythms = rhythms
        self.pitch_pool = sorted(set(pitches))
        self.rhythm_pool = sorted(set(rhythms))
        random.seed(50)
    
    def generate_melody(self, length=16):
        """
        Generate a random melody by selecting notes and rhythms by chance.
        
        Args:
            length: Number of notes to generate (default 16)
        
        Returns:
            List of [pitch, duration] pairs
        """
        melody = []
        for _ in range(length):
            pitch = random.choice(self.pitch_pool)
            rhythm = random.choice(self.rhythm_pool)
            melody.append([pitch, rhythm])
        return melody
