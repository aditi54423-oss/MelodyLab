import random


class FirstOrderMarkovGenerator:
    """First-Order Markov melody generator - remembers the previous note event."""

    def __init__(self, pitches, rhythms):
        """
        Initialize First-Order Markov melody generator.

        Args:
            pitches: List of training pitches
            rhythms: List of training rhythms (durations)
        """
        self.pitches = pitches
        self.rhythms = rhythms
        self.events = list(zip(pitches, rhythms))
        self.event_chain = self._build_first_order_chain(self.events)
        random.seed(50)

    def _build_first_order_chain(self, seq):
        """
        Construct a first-order Markov chain from a sequence.
        Returns a dictionary mapping each event to a list of (next_event, probability).
        """
        trans = {}
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            trans.setdefault(a, []).append(b)

        chain = {}
        for a, lst in trans.items():
            freq = {}
            for x in lst:
                freq[x] = freq.get(x, 0) + 1
            s = sum(freq.values())
            chain[a] = [(k, freq[k] / s) for k in freq]
        return chain

    def _generate_sequence(self, chain, length, start_state=None):
        """
        Generate a sequence from a first-order Markov chain.

        Args:
            chain: The Markov chain dictionary
            length: Length of sequence to generate
            start_state: Starting event (if valid in chain)

        Returns:
            Tuple of (sequence, restart_count)
        """
        keys = list(chain.keys())
        if not keys:
            return random.choices(self.events, k=length), 0

        curr = start_state if (start_state in chain) else random.choice(keys)
        out = [curr]
        restart_count = 0

        for _ in range(length - 1):
            nxts = chain.get(curr)
            if not nxts:
                curr = random.choice(keys)
                restart_count += 1
            else:
                choices, probs = zip(*nxts)
                curr = random.choices(choices, probs)[0]
            out.append(curr)

        return out, restart_count

    def generate_melody(self, length=16):
        """
        Generate a melody using a first-order Markov chain over note events.

        Args:
            length: Number of notes to generate (default 16)

        Returns:
            List of [pitch, duration] pairs
        """
        start_event = random.choice(self.events)
        event_seq, _ = self._generate_sequence(self.event_chain, length, start_event)
        return [[pitch, rhythm] for pitch, rhythm in event_seq]
