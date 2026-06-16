import random


class SecondOrderMarkovGenerator:
    """Second-Order Markov melody generator - remembers the previous two note events."""

    def __init__(self, pitches, rhythms):
        """
        Initialize Second-Order Markov melody generator.

        Args:
            pitches: List of training pitches
            rhythms: List of training rhythms (durations)
        """
        self.pitches = pitches
        self.rhythms = rhythms
        self.events = list(zip(pitches, rhythms))
        self.event_chain_2 = self._build_second_order_chain(self.events)
        self.event_chain_1 = self._build_first_order_chain(self.events)
        self.event_pool, self.event_weights = self._build_weighted_pool(self.events)
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

    def _build_second_order_chain(self, seq):
        """
        Construct a second-order Markov chain from a sequence.
        Returns a dictionary mapping each pair of consecutive events to a list of
        (next_event, probability).
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
            chain[pair] = [(k, freq[k] / s) for k in freq]
        return chain

    def _build_weighted_pool(self, seq):
        """Build weighted fallback probabilities from full note-event frequencies."""
        freq = {}
        for item in seq:
            freq[item] = freq.get(item, 0) + 1

        total = sum(freq.values())
        pool = list(freq.keys())
        weights = [freq[item] / total for item in pool]

        return pool, weights

    def _choose_weighted_event(self):
        """Choose one full note event using training-event frequencies."""
        return random.choices(self.event_pool, weights=self.event_weights)[0]

    def _generate_sequence(self, length, starting_pair=None):
        """
        Generate a sequence from a second-order event Markov chain.

        Backoff strategy:
        1. Try second-order event memory: (event1, event2) -> next event
        2. If missing, back off to first-order event memory: event2 -> next event
        3. If still missing, choose a weighted-random training event

        Returns:
            Tuple of (sequence, fallback_count)
        """
        if length <= 0:
            return [], 0

        if length == 1:
            return [self._choose_weighted_event()], 0

        pair_keys = list(self.event_chain_2.keys())
        if starting_pair in self.event_chain_2:
            curr = starting_pair
        elif pair_keys:
            curr = random.choice(pair_keys)
        elif len(self.events) >= 2:
            start_index = random.randint(0, len(self.events) - 2)
            curr = (self.events[start_index], self.events[start_index + 1])
        else:
            event = self._choose_weighted_event()
            curr = (event, event)

        out = [curr[0], curr[1]]
        fallback_count = 0

        for _ in range(length - 2):
            second_order_choices = self.event_chain_2.get(curr)

            if second_order_choices:
                choices, probs = zip(*second_order_choices)
                nxt = random.choices(choices, probs)[0]
            else:
                last_event = curr[1]
                first_order_choices = self.event_chain_1.get(last_event)
                fallback_count += 1

                if first_order_choices:
                    choices, probs = zip(*first_order_choices)
                    nxt = random.choices(choices, probs)[0]
                else:
                    nxt = self._choose_weighted_event()

            out.append(nxt)
            curr = (curr[1], nxt)

        return out, fallback_count

    def generate_melody(self, length=16):
        """
        Generate a melody using a second-order Markov chain over note events.

        Args:
            length: Number of notes to generate (default 16)

        Returns:
            List of [pitch, duration] pairs
        """
        if len(self.events) >= 2:
            start_index = random.randint(0, len(self.events) - 2)
            starting_pair = (self.events[start_index], self.events[start_index + 1])
        else:
            starting_pair = None

        event_seq, _ = self._generate_sequence(length, starting_pair)
        return [[pitch, rhythm] for pitch, rhythm in event_seq]
