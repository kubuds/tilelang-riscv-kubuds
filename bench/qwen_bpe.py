"""Qwen2/3 byte-level BPE encoder/decoder using GGUF tokenizer metadata."""
import re


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class QwenBPE:
    def __init__(self, tokens, merges):
        self.tokens = tokens
        self.vocab = {t: i for i, t in enumerate(tokens)}
        self.ranks = {}
        for i, m in enumerate(merges):
            a, b = m.split(" ", 1)
            self.ranks[(a, b)] = i
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+", re.IGNORECASE)

    def _bpe(self, word):
        parts = list(word)
        while len(parts) > 1:
            best = None; br = None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (br is None or r < br):
                    br = r; best = i
            if best is None:
                break
            a, b = parts[best], parts[best + 1]
            out = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(parts[i]); i += 1
            parts = out
        return parts

    def encode(self, text):
        ids = []
        for piece in self.pat.findall(text):
            word = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(word):
                ids.append(self.vocab[tok])
        return ids

    def decode(self, ids):
        s = "".join(self.tokens[i] for i in ids)
        raw = bytes(self.u2b.get(ch, 0) for ch in s)
        return raw.decode("utf-8", "replace")
