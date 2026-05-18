import random

class FirstOrderMarkovGenerator:
    """First-Order Markov melody generator - remembers the previous note."""
    
    def __init__(self, pitches, rhythms):
        """
        Initialize First-Order Markov melody generator.
        
        Args:
            pitches: List of training pitches
            rhythms: List of training rhythms (durations)
        """
        self.pitches = pitches
        self.rhythms = rhythms
        self.note_chain = self._build_first_order_chain(pitches)
        self.rhythm_chain = self._build_first_order_chain(rhythms)
        random.seed(50)
    
    def _build_first_order_chain(self, seq):
        """
        Construct a first-order Markov chain from a sequence.
        Returns a dictionary mapping each element to a list of (next_element, probability).
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
            # Normalize counts to probabilities
            chain[a] = [(k, freq[k] / s) for k in freq]
        return chain
    
    def _generate_sequence(self, chain, length, start_state=None):
        """
        Generate a sequence from a first-order Markov chain.
        
        Args:
            chain: The Markov chain dictionary
            length: Length of sequence to generate
            start_state: Starting state (if valid in chain)
        
        Returns:
            Tuple of (sequence, restart_count)
        """
        keys = list(chain.keys())
        curr = start_state if (start_state in chain) else random.choice(keys)
        out = [curr]
        restart_count = 0
        
        for _ in range(length - 1):
            nxts = chain.get(curr)
            if not nxts:
                # Fallback: restart from random valid state
                curr = random.choice(keys)
                restart_count += 1
            else:
                choices, probs = zip(*nxts)
                curr = random.choices(choices, probs)[0]
            out.append(curr)
        
        return out, restart_count
    
    def generate_melody(self, length=16):
        """
        Generate a melody using first-order Markov chains.
        
        Args:
            length: Number of notes to generate (default 16)
        
        Returns:
            List of [pitch, duration] pairs
        """
        n_start = random.choice(self.pitches)
        r_start = random.choice(self.rhythms)
        
        n_seq, _ = self._generate_sequence(self.note_chain, length, n_start)
        r_seq, _ = self._generate_sequence(self.rhythm_chain, length, r_start)
        
        # Zip pitches and rhythms together
        melody = [[n, r] for n, r in zip(n_seq, r_seq)]
        return melody
