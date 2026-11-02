"""Qwen2/3 byte-level BPE encoder/decoder using GGUF tokenizer metadata."""
import re
import unicodedata


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
        self.pat = None

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

    @staticmethod
    def _cls(ch):
        cat = unicodedata.category(ch)
        if cat.startswith("L"): return "L"
        if cat.startswith("N"): return "N"
        if ch in "\r\n": return "Nl"
        if ch.isspace(): return "S"
        return "P"

    def _split(self, text):
        out = []
        i, n = 0, len(text)
        while i < n:
            c = text[i]
            # contractions
            low = text[i:i+3].lower()
            hit = None
            for t in ("'s", "'t", "'m", "'d", "'re", "'ve", "'ll"):
                if low.startswith(t):
                    hit = t; break
            if hit:
                out.append(text[i:i+len(hit)]); i += len(hit); continue
            pref = ""
            j = i
            if c not in "\r\n" and self._cls(c) not in ("L", "N"):
                pref = c; j = i + 1
            if j < n and self._cls(text[j]) == "L":
                k = j
                while k < n and self._cls(text[k]) == "L": k += 1
                out.append(pref + text[j:k]); i = k; continue
            if self._cls(c) == "N":
                out.append(c); i += 1; continue
            j = i; pref = ""
            if c == " " and i + 1 < n and self._cls(text[i+1]) == "P":
                pref = " "; j = i + 1
            if j < n and self._cls(text[j]) == "P":
                k = j
                while k < n and self._cls(text[k]) == "P": k += 1
                while k < n and text[k] in "\r\n": k += 1
                out.append(pref + text[j:k]); i = k; continue
            if self._cls(c) == "S":
                k = i
                while k < n and self._cls(text[k]) == "S": k += 1
                run = text[i:k]
                if "\n" in run or "\r" in run:
                    buf = ""
                    for ch in run:
                        buf += ch
                        if ch in "\r\n":
                            out.append(buf); buf = ""
                    if buf: out.append(buf)
                else:
                    out.append(run)
                i = k; continue
            out.append(c); i += 1
        return out

    def encode(self, text):
        ids = []
        for piece in self._split(text):
            word = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(word):
                ids.append(self.vocab[tok])
        return ids

    def decode(self, ids):
        s = "".join(self.tokens[i] for i in ids)
        raw = bytes(self.u2b.get(ch, 0) for ch in s)
        return raw.decode("utf-8", "replace")
