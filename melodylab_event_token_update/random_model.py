import random


class RandomMelodyGenerator:
    """Random melody generator - chooses full note events by chance."""

    def __init__(self, pitches, rhythms):
        """
        Initialize Random melody generator.

        Args:
            pitches: List of available training pitches
            rhythms: List of available training rhythms (durations)
        """
        self.pitches = pitches
        self.rhythms = rhythms
        self.events = list(zip(pitches, rhythms))
        self.event_pool = sorted(set(self.events))
        random.seed(50)

    def generate_melody(self, length=16):
        """
        Generate a random melody by selecting full pitch-rhythm events by chance.

        Args:
            length: Number of notes to generate (default 16)

        Returns:
            List of [pitch, duration] pairs
        """
        melody = []
        for _ in range(length):
            pitch, rhythm = random.choice(self.event_pool)
            melody.append([pitch, rhythm])
        return melody
