import numpy as np
import random
from typing import Optional

class Language:
    def __init__(self):
        self.symbols = {}               # symbol -> meaning_vector
        self.grammar_patterns = []      # unused for now
        self.utterances = []            # history of all utterances
        self.symbol_counter = 0

    def create_symbol(self, meaning_vector: np.ndarray) -> str:
        """Create a new pseudorandom symbol for a concept."""
        vowels    = "aeiou"
        consonants= "bcdfghjklmnpqrstvwxyz"
        length    = random.randint(2,6)
        s = ""
        for i in range(length):
            s += consonants if i%2==0 else vowels
        if s in self.symbols:
            s += str(self.symbol_counter)
            self.symbol_counter += 1
        self.symbols[s] = meaning_vector
        return s

    def find_symbol(self, meaning_vector: np.ndarray, threshold: float = 0.8) -> Optional[str]:
        """Return an existing symbol if its meaning‐vector is similar enough."""
        for sym, vec in self.symbols.items():
            sim = np.dot(meaning_vector, vec) / (np.linalg.norm(meaning_vector)*np.linalg.norm(vec))
            if sim > threshold:
                return sym
        return None

    def mutate_symbol(self, symbol: str) -> str:
        """Slight random variation of an existing symbol."""
        if symbol not in self.symbols:
            return symbol
        chars = list(symbol)
        if random.random() < 0.5 and len(chars) > 2:
            i, j = random.sample(range(len(chars)), 2)
            chars[i], chars[j] = chars[j], chars[i]
        else:
            i = random.randrange(len(chars))
            chars[i] = random.choice("aeiou" if chars[i] in "aeiou" else "bcdfghjklmnpqrstvwxyz")
        new_sym = "".join(chars)
        # mutate its meaning-embedding slightly
        self.symbols[new_sym] = self.symbols[symbol] + np.random.normal(0, 0.1, len(self.symbols[symbol]))
        return new_sym
