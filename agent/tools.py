"""Offline Search/Lookup tools over a question's HotpotQA distractor paragraphs.

Using the per-question paragraph pool (gold + distractors) keeps runs
deterministic and free of Wikipedia API drift. Search returns the best-matching
paragraph; Lookup[kw] then returns successive sentences of that paragraph
containing kw, mirroring the ReAct paper's Search/Lookup interface.
"""
import re


def _tok(s):
    return set(re.findall(r"\w+", s.lower()))


class ParagraphSearch:
    def __init__(self, titles, sentences):
        self.docs = {t: " ".join(s) for t, s in zip(titles, sentences)}
        self.sents = {t: [x.strip() for x in s if x.strip()] for t, s in zip(titles, sentences)}
        self.current = None
        self._kw, self._idx = None, 0

    def search(self, query, k=1):
        q = _tok(query)
        scored = []
        for title, text in self.docs.items():
            score = 2 * len(q & _tok(title)) + len(q & _tok(text)) / (1 + len(_tok(text)) ** 0.5)
            scored.append((score, title))
        scored.sort(reverse=True)
        top = [t for s, t in scored[:k] if s > 0]
        if not top:
            return "No results found."
        self.current, self._kw, self._idx = top[0], None, 0
        return "\n".join(f"[{t}] {self.docs[t]}" for t in top)

    def lookup(self, keyword):
        if self.current is None:
            return "The last page Searched was not found, so you cannot Lookup a keyword in it."
        hits = [s for s in self.sents[self.current] if keyword.lower() in s.lower()]
        if keyword != self._kw:
            self._kw, self._idx = keyword, 0
        if not hits:
            return "No Results"
        if self._idx >= len(hits):
            return "No more results.\n"
        out = f"(Result {self._idx + 1} / {len(hits)}) {hits[self._idx]}"
        self._idx += 1
        return out
