import random

class SecondOrderMarkovGenerator:
    """Second-Order Markov melody generator - remembers the previous two notes."""
    
    def __init__(self, pitches, rhythms):
        """
        Initialize Second-Order Markov melody generator.
        
        Args:
            pitches: List of training pitches
            rhythms: List of training rhythms (durations)
        """
        self.pitches = pitches
        self.rhythms = rhythms
        self.note_chain = self._build_second_order_chain(pitches)
        self.rhythm_chain = self._build_second_order_chain(rhythms)
        random.seed(50)
    
    def _build_second_order_chain(self, seq):
        """
        Construct a second-order Markov chain from a sequence.
        Returns a dictionary mapping each pair of consecutive elements to a list of (next_element, probability).
        """
        trans = {}
        for i in range(len(seq) - 2):
            pair = (seq[i], seq[i + 1])
            nxt = seq[i + 2]
            trans.setdefault(pair, []).append(nxt)
        
        chain = {}
        for pair, lst in trans.items():
            freq = {}
            for x in lst:
                freq[x] = freq.get(x, 0) + 1
            s = sum(freq.values())
            # Normalize counts to probabilities
            chain[pair] = [(k, freq[k] / s) for k in freq]
        return chain
    
    def _generate_sequence(self, chain, length, starting_pair=None):
        """
        Generate a sequence from a second-order Markov chain.
        
        Args:
            chain: The Markov chain dictionary
            length: Length of sequence to generate
            starting_pair: Starting pair of states (if valid in chain)
        
        Returns:
            Tuple of (sequence, restart_count)
        """
        keys = list(chain.keys())
        curr = starting_pair if (starting_pair in chain) else random.choice(keys)
        out = [curr[0], curr[1]]
        restart_count = 0
        
        for _ in range(length - 2):
            nxts = chain.get(curr)
            if not nxts:
                # Fallback: restart from random valid state pair
                curr = random.choice(keys)
                out.append(curr[1])
                restart_count += 1
            else:
                choices, probs = zip(*nxts)
                nxt = random.choices(choices, probs)[0]
                out.append(nxt)
                # Shift pair window for next iteration
                curr = (curr[1], nxt)
        
        return out, restart_count
    
    def generate_melody(self, length=16):
        """
        Generate a melody using second-order Markov chains.
        
        Args:
            length: Number of notes to generate (default 16)
        
        Returns:
            List of [pitch, duration] pairs
        """
        # Pick random starting indices for 2-note / 2-duration seeds
        iN = random.randint(0, len(self.pitches) - 2)
        iR = random.randint(0, len(self.rhythms) - 2)
        n_start = (self.pitches[iN], self.pitches[iN + 1])
        r_start = (self.rhythms[iR], self.rhythms[iR + 1])
        
        # Generate sequences using second-order Markov chain
        n_seq, _ = self._generate_sequence(self.note_chain, length, n_start)
        r_seq, _ = self._generate_sequence(self.rhythm_chain, length, r_start)
        
        # Zip pitches and rhythms together
        melody = [[n, r] for n, r in zip(n_seq, r_seq)]
        return melody
