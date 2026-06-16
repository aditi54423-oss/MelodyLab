import random


class WeightedRandomMelodyGenerator:
    """Weighted Random melody generator - chooses common training note events more often."""

    def __init__(self, pitches, rhythms):
        """
        Initialize Weighted Random melody generator.

        Args:
            pitches: List of training pitches
            rhythms: List of training rhythms (durations)
        """
        self.pitches = pitches
        self.rhythms = rhythms
        self.events = list(zip(pitches, rhythms))
        self.event_pool, self.event_weights = self._build_weighted_pool(self.events)
        random.seed(50)

    def _build_weighted_pool(self, seq):
        """
        Build a weighted pool from a sequence.
        Returns a tuple of:
        - unique items
        - matching probabilities based on frequency
        """
        freq = {}
        for item in seq:
            freq[item] = freq.get(item, 0) + 1

        total = sum(freq.values())
        pool = list(freq.keys())
        weights = [freq[item] / total for item in pool]

        return pool, weights

    def generate_melody(self, length=16):
        """
        Generate a weighted-random melody.

        Args:
            length: Number of notes to generate

        Returns:
            List of [pitch, duration] pairs
        """
        melody = []

        for _ in range(length):
            pitch, rhythm = random.choices(
                self.event_pool,
                weights=self.event_weights,
            )[0]
            melody.append([pitch, rhythm])

        return melody
