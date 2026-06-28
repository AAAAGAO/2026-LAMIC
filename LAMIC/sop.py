from __future__ import annotations

import re
from dataclasses import dataclass


CODE_BLOCK_PATTERN = re.compile(r"```.*?```|`[^`]+`", re.DOTALL)
MULTILINE_CODE_PATTERN = re.compile(r"(?m)(?:^(?:\s{4,}|\t).+\n?){2,}")


@dataclass(slots=True)
class SOPClause:
    action: str
    object_text: str
    purpose: str
    usage_role: str


class SOPExtractor:
    def __init__(self, model_name: str = "en_core_web_trf", max_clauses: int = 6) -> None:
        self.model_name = model_name
        self.max_clauses = max_clauses
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy

            try:
                self._nlp = spacy.load(self.model_name)
            except OSError as exc:
                raise RuntimeError(
                    f"spaCy model '{self.model_name}' is not installed.\n"
                    f"Install it with:\n"
                    f"python -m spacy download {self.model_name}\n"
                    f"Or switch config.model.spacy_model_name to an installed model."
                ) from exc
        return self._nlp

    def preprocess(self, fragment: str) -> str:
        replaced = CODE_BLOCK_PATTERN.sub(" <CODE_BLOCK> ", fragment)
        replaced = MULTILINE_CODE_PATTERN.sub(" <CODE_BLOCK> ", replaced)
        return re.sub(r"\s+", " ", replaced).strip()

    def infer_usage_role(self, sentence: str) -> str:
        lowered = sentence.lower()
        if any(token in lowered for token in ["error", "exception", "fix", "issue", "problem", "debug"]):
            return "troubleshooting"
        if any(token in lowered for token in ["must", "cannot", "can't", "should not", "only if", "limit"]):
            return "constraint"
        if any(token in lowered for token in ["for example", "for instance", "e.g.", "example"]):
            return "example_only"
        if any(token in lowered for token in ["see", "refer to", "documentation", "javadoc", "reference"]):
            return "reference_only"
        if sentence.count(",") >= 2 and all(mark not in lowered for mark in ["use", "call", "create", "set"]):
            return "list_only"
        if any(token in lowered for token in ["represented by", "extends", "implements", "consists of"]):
            return "static_relation"
        if any(token in lowered for token in ["use", "call", "create", "set", "get", "convert", "parse", "retrieve"]):
            return "actual_use"
        return "mention_only"

    def extract_clause(self, sent) -> SOPClause:
        root = sent.root
        action = root.lemma_.lower() if root.pos_ in {"VERB", "AUX"} else "mention"
        object_text = ""
        purpose = "overview"

        for child in root.children:
            if child.dep_ in {"dobj", "obj", "attr", "oprd"} and not object_text:
                object_text = child.text
            elif child.dep_ == "prep":
                pobj = next((token for token in child.children if token.dep_ == "pobj"), None)
                if pobj is not None and purpose == "overview":
                    purpose = f"{child.text} {pobj.text}".strip()
            elif child.dep_ in {"xcomp", "ccomp"} and purpose == "overview":
                purpose = child.text

        if not object_text:
            noun_chunks = list(sent.noun_chunks)
            if noun_chunks:
                object_text = noun_chunks[0].text

        if purpose == "overview":
            lowered = sent.text.lower()
            if " to " in lowered:
                purpose = lowered.split(" to ", 1)[1][:60].strip(" .;")
            elif " for " in lowered:
                purpose = lowered.split(" for ", 1)[1][:60].strip(" .;")

        usage_role = self.infer_usage_role(sent.text)
        return SOPClause(
            action=action or "mention",
            object_text=object_text or "unknown",
            purpose=purpose or "overview",
            usage_role=usage_role,
        )

    def extract(self, fragment: str, api: str) -> str:
        text = self.preprocess(fragment)
        doc = self.nlp(text)
        clauses: list[str] = []
        for sent in doc.sents:
            clause = self.extract_clause(sent)
            clauses.append(
                f"API={api} ; ACTION={clause.action} ; OBJECT={clause.object_text} ; "
                f"PURPOSE={clause.purpose} ; USAGE_ROLE={clause.usage_role}"
            )
            if len(clauses) >= self.max_clauses:
                break
        if not clauses:
            clauses.append(
                f"API={api} ; ACTION=mention ; OBJECT=unknown ; PURPOSE=overview ; USAGE_ROLE=mention_only"
            )
        return " || ".join(clauses)
