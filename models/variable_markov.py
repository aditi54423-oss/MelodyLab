import random


class VariableMarkovGenerator:
    """Variable-Order Markov melody generator - remembers up to five previous note events."""

    def __init__(self, pitches, rhythms, max_order=5, min_count_by_order=None):
        """
        Initialize Variable-Order Markov melody generator.

        Args:
            pitches: List of training pitches
            rhythms: List of training rhythms (durations)
            max_order: Maximum number of previous note events to remember
            min_count_by_order: Optional dictionary controlling how many times
                a pattern must appear before the model trusts it.
        """
        self.pitches = pitches
        self.rhythms = rhythms
        self.events = list(zip(pitches, rhythms))
        self.max_order = max_order

        if min_count_by_order is None:
            self.min_count_by_order = {
                5: 3,
                4: 3,
                3: 2,
                2: 1,
                1: 1,
            }
        else:
            self.min_count_by_order = min_count_by_order

        self.event_chains = self._build_variable_order_chains(self.events)
        self.event_pool, self.event_weights = self._build_weighted_pool(self.events)
        self.last_trace = []
        random.seed(50)

    def _build_chain_for_order(self, seq, order):
        """
        Construct one Markov chain for a specific order.

        Returns a dictionary mapping a context tuple to a list of
        (next_event, probability, count) tuples.
        """
        trans = {}

        for i in range(len(seq) - order):
            context = tuple(seq[i:i + order])
            nxt = seq[i + order]
            trans.setdefault(context, []).append(nxt)

        chain = {}
        for context, lst in trans.items():
            freq = {}
            for item in lst:
                freq[item] = freq.get(item, 0) + 1

            total = sum(freq.values())
            chain[context] = [
                (item, freq[item] / total, freq[item])
                for item in freq
            ]

        return chain

    def _build_variable_order_chains(self, seq):
        """
        Build Markov chains for order 1 up to max_order.
        """
        chains = {}

        for order in range(1, self.max_order + 1):
            chains[order] = self._build_chain_for_order(seq, order)

        return chains

    def _build_weighted_pool(self, seq):
        """
        Build weighted fallback probabilities from full note-event frequencies.
        """
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

    def _format_event(self, event):
        """Convert an event tuple into an app-friendly dictionary."""
        pitch, rhythm = event
        return {
            "pitch": pitch,
            "rhythm": rhythm,
        }

    def _format_context(self, context):
        """Convert a context tuple into an app-friendly list."""
        return [self._format_event(event) for event in context]

    def _format_options(self, options, limit=5):
        """
        Convert chain options into app-friendly candidate dictionaries.
        options are stored as (event, probability, count).
        """
        sorted_options = sorted(options, key=lambda item: item[1], reverse=True)
        formatted = []

        for event, probability, count in sorted_options[:limit]:
            pitch, rhythm = event
            formatted.append({
                "pitch": pitch,
                "rhythm": rhythm,
                "probability": round(probability, 4),
                "percentage": round(probability * 100, 1),
                "count": count,
            })

        return formatted

    def _choose_next_event(self, melody_so_far, return_trace=True):
        """
        Choose the next event using variable-order backoff.

        Strategy:
        1. Try the longest available memory, up to max_order.
        2. If the context is missing or too rare, try a shorter memory.
        3. If all contexts fail, use weighted-random fallback.

        Returns:
            Tuple of (next_event, trace_info)
        """
        trace_attempts = []
        longest_possible_order = min(self.max_order, len(melody_so_far))

        for order in range(longest_possible_order, 0, -1):
            context = tuple(melody_so_far[-order:])
            choices = self.event_chains.get(order, {}).get(context)
            min_count = self.min_count_by_order.get(order, 1)

            if not choices:
                trace_attempts.append({
                    "order": order,
                    "context": self._format_context(context),
                    "status": "missing",
                    "message": "This pattern was not found in the training data.",
                    "total_count": 0,
                    "min_count": min_count,
                    "possible_next_events": [],
                })
                continue

            total_count = sum(count for _, _, count in choices)

            if total_count < min_count:
                trace_attempts.append({
                    "order": order,
                    "context": self._format_context(context),
                    "status": "too_rare",
                    "message": "This pattern was found, but not often enough to trust.",
                    "total_count": total_count,
                    "min_count": min_count,
                    "possible_next_events": self._format_options(choices),
                })
                continue

            events, probabilities, _counts = zip(*choices)
            selected_event = random.choices(events, probabilities)[0]
            selected_pitch, selected_rhythm = selected_event

            trace_attempts.append({
                "order": order,
                "context": self._format_context(context),
                "status": "used",
                "message": "This was the longest reliable pattern found.",
                "total_count": total_count,
                "min_count": min_count,
                "possible_next_events": self._format_options(choices),
            })

            trace_info = {
                "used_order": order,
                "used_context": self._format_context(context),
                "attempts": trace_attempts,
                "possible_next_events": self._format_options(choices),
                "selected_event": self._format_event(selected_event),
                "selected_note": selected_pitch,
                "selected_rhythm": selected_rhythm,
                "fallback_used": False,
                "explanation": (
                    f"The model used a reliable {order}-event memory and selected "
                    f"{selected_pitch} with duration {selected_rhythm}."
                ),
            }

            return selected_event, trace_info

        selected_event = self._choose_weighted_event()
        selected_pitch, selected_rhythm = selected_event

        trace_info = {
            "used_order": 0,
            "used_context": [],
            "attempts": trace_attempts,
            "possible_next_events": [],
            "selected_event": self._format_event(selected_event),
            "selected_note": selected_pitch,
            "selected_rhythm": selected_rhythm,
            "fallback_used": True,
            "explanation": (
                f"No reliable Markov pattern was found, so the model used weighted random fallback "
                f"and selected {selected_pitch} with duration {selected_rhythm}."
            ),
        }

        return selected_event, trace_info

    def _choose_start_sequence(self):
        """
        Choose a starting sequence from the training events.
        """
        if not self.events:
            return []

        start_length = min(self.max_order, len(self.events))

        if len(self.events) <= start_length:
            return self.events[:]

        start_index = random.randint(0, len(self.events) - start_length)
        return self.events[start_index:start_index + start_length]

    def _generate_sequence(self, length):
        """
        Generate a sequence using variable-order Markov backoff.

        Returns:
            Tuple of (sequence, fallback_count, trace)
        """
        if length <= 0:
            return [], 0, []

        if not self.events:
            return [], 0, []

        melody = self._choose_start_sequence()
        melody = melody[:length]
        trace = []
        fallback_count = 0

        while len(melody) < length:
            melody_before_choice = melody[:]
            next_event, trace_info = self._choose_next_event(melody_before_choice)

            if trace_info["fallback_used"]:
                fallback_count += 1

            melody.append(next_event)

            trace_info["step"] = len(melody)
            trace_info["melody_so_far"] = [
                self._format_event(event) for event in melody_before_choice
            ]
            trace.append(trace_info)

        return melody, fallback_count, trace

    def generate_melody(self, length=16):
        """
        Generate a melody using a variable-order Markov chain over note events.

        Args:
            length: Number of notes to generate (default 16)

        Returns:
            List of [pitch, duration] pairs
        """
        event_seq, _fallback_count, trace = self._generate_sequence(length)
        self.last_trace = trace
        return [[pitch, rhythm] for pitch, rhythm in event_seq]

    def get_trace_for_step(self, step):
        """
        Return trace information for a generated melody step.

        step is zero-based. The starting notes are chosen directly from the
        training data, so they may not have candidate-selection trace entries.
        """
        for item in self.last_trace:
            if item.get("step") == step + 1:
                return item

        return None
