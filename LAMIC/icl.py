from __future__ import annotations

import json
import os
import random
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .clues import (
    build_sample_clue_features,
    extract_answer_focused_excerpt,
    infer_sample_decision_profile,
    render_demo_reason,
    render_sample_clue_text,
)
from .config import ICLConfig
from .data import ApiSample
from .feedback import should_verify_so_positive
from .reason_feedback import build_focus_fragment
from .retrieval import RetrievalRow


SYSTEM_PROMPT = """You are an API knowledge identification assistant.
Determine whether the given knowledge unit is relevant to the target API.
Relevant means the knowledge unit explains how the API is used, what role it plays, or how it helps solve an API-related task.
Return the final decision inside <LABEL> and </LABEL> as Relevant or Irrelevant."""

ENHANCEMENT_SYSTEM_PROMPT = """You generate clue-augmented demonstrations for API knowledge location.
Given a labeled <API, KU> example, output JSON with keys clues and reasoning.
clues must summarize local evidence that supports the gold label.
reasoning must briefly explain why the clues support that label."""

SO_POSITIVE_PROFILES = {"so_solution_with_target_api", "so_concise_api_recipe"}
SO_NEGATIVE_PROFILES = {
    "so_unresolved_question",
    "so_wrong_api_focus",
    "so_solution_but_target_unclear",
    "so_mixed_or_weak_qa",
}


def _is_android_sample(sample: ApiSample) -> bool:
    return sample.language == "android"


@dataclass(slots=True)
class Prediction:
    label: int
    reason: str
    raw_response: str
    llm_label: int
    calibration_applied: bool = False
    calibration_note: str = ""


@dataclass(slots=True)
class DemoAugmentation:
    clues: str
    reasoning: str
    raw_response: str = ""
    generated_by_llm: bool = False


class DeepSeekClient:
    def __init__(self, config: ICLConfig) -> None:
        self.api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key is required. Pass --api-key or set DEEPSEEK_API_KEY.")
        self.model_name = config.model_name
        self.url = config.url
        self.timeout_seconds = config.timeout_seconds
        self.max_retries = config.max_retries
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def chat(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.0,
                    },
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                content = str(data["choices"][0]["message"]["content"])
                if not content.strip():
                    raise RuntimeError("DeepSeek returned an empty response")
                return content
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"DeepSeek request failed after {self.max_retries} attempts") from last_error


def fallback_demo_augmentation(sample: ApiSample) -> DemoAugmentation:
    return DemoAugmentation(
        clues=render_sample_clue_text(sample),
        reasoning=render_demo_reason(sample),
        generated_by_llm=False,
    )


def _label_name(label: int) -> str:
    return "Relevant" if int(label) == 1 else "Irrelevant"


def build_demo_enhancement_prompt(sample: ApiSample) -> str:
    return "\n\n".join(
        [
            "Create clue-augmented reasoning for this labeled demonstration.",
            "Focus on whether the target API itself is the main explained or solved API, not merely a nearby class, return type, example value, or list item.",
            "For StackOverflow, separate the question text from the answer/fix. A useful answer for another API should not make the target API relevant.",
            "For tutorials, distinguish substantive usage/role explanation from API lists, implementation inventories, and Javadoc-like structural facts.",
            f"API: {sample.api}",
            f"Gold Label: {_label_name(sample.label)}",
            f"Knowledge Unit:\n{sample.fragment[:2800]}",
            (
                "Output JSON only with keys clues and reasoning. "
                "clues should name decisive evidence such as target_api_focus, answer_or_tutorial_focus, "
                "usage_or_role_evidence, and incidental_or_list_evidence."
            ),
        ]
    )


def parse_demo_augmentation(raw_response: str) -> DemoAugmentation:
    start = raw_response.find("{")
    end = raw_response.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Enhancement response is not JSON: {raw_response}")
    payload = json.loads(raw_response[start : end + 1])
    raw_clues = payload.get("clues", "")
    if isinstance(raw_clues, list):
        clues = "; ".join(str(item).strip() for item in raw_clues if str(item).strip())
    else:
        clues = str(raw_clues).strip()
    reasoning = str(payload.get("reasoning", "")).strip()
    if not clues or not reasoning:
        raise ValueError(f"Enhancement response misses clues or reasoning: {raw_response}")
    return DemoAugmentation(
        clues=clues,
        reasoning=reasoning,
        raw_response=raw_response,
        generated_by_llm=True,
    )


class ClueBasedDemoEnhancer:
    def __init__(self, client: DeepSeekClient, cache_path: str | Path | None = None) -> None:
        self.client = client
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict[str, str | bool]] = {}
        self._lock = threading.Lock()
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.cache = {}

    def _cache_key(self, sample: ApiSample) -> str:
        return f"{sample.sample_id}:{sample.label}"

    def enhance(self, sample: ApiSample) -> DemoAugmentation:
        key = self._cache_key(sample)
        with self._lock:
            cached = self.cache.get(key)
        if cached:
            return DemoAugmentation(
                clues=str(cached.get("clues", "")),
                reasoning=str(cached.get("reasoning", "")),
                raw_response=str(cached.get("raw_response", "")),
                generated_by_llm=bool(cached.get("generated_by_llm", True)),
            )
        try:
            augmentation = parse_demo_augmentation(
                self.client.chat(
                    build_demo_enhancement_prompt(sample),
                    system_prompt=ENHANCEMENT_SYSTEM_PROMPT,
                )
            )
        except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
            fallback = fallback_demo_augmentation(sample)
            augmentation = DemoAugmentation(
                clues=fallback.clues,
                reasoning=f"{fallback.reasoning} LLM clue enhancement failed: {exc}",
                raw_response="",
                generated_by_llm=False,
            )
        with self._lock:
            self.cache[key] = {
                "clues": augmentation.clues,
                "reasoning": augmentation.reasoning,
                "raw_response": augmentation.raw_response,
                "generated_by_llm": augmentation.generated_by_llm,
            }
        return augmentation

    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = dict(self.cache)
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
        last_error: OSError | None = None
        for attempt in range(3):
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    delete=False,
                    dir=self.cache_path.parent,
                    prefix=f"{self.cache_path.stem}.",
                    suffix=".tmp",
                ) as handle:
                    handle.write(serialized)
                    temp_path = Path(handle.name)
                temp_path.replace(self.cache_path)
                return
            except OSError as exc:
                last_error = exc
                try:
                    if "temp_path" in locals() and temp_path.exists():
                        temp_path.unlink()
                except OSError:
                    pass
                time.sleep(0.5 * (attempt + 1))
        raise OSError(f"Failed to save demo enhancement cache: {self.cache_path}") from last_error


def _source_name(source: str) -> str:
    return {
        "SO": "StackOverflow",
        "TU": "Tutorial",
    }.get(source, source)


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def _normalized_fragment(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_question_like(fragment: str) -> bool:
    return _contains_any(
        fragment,
        (
            "?",
            "how do i",
            "how can i",
            "any ideas",
            "what should i use",
            "why am i",
            "i have this problem",
            "it throws",
            "exception",
            "error",
            "help",
        ),
    )


def _is_solution_like(fragment: str) -> bool:
    return _contains_any(
        fragment,
        (
            "try the following",
            "you can use",
            "the solution",
            "workaround",
            "use instead",
            "for example",
            "here's",
            "this works",
            "returns",
            "correct way",
            "accepted answer",
        ),
    )


def _is_list_like_tutorial(fragment: str) -> bool:
    return _contains_any(
        fragment,
        (
            "complete list",
            "implementations are",
            "other implementations include",
            "for instance",
            "is represented by",
            "the following values",
        ),
    )


def _ordered_rows(retrieved_rows: list[RetrievalRow], config: ICLConfig) -> list[RetrievalRow]:
    demos = list(retrieved_rows)
    if config.order_strategy == "nearest_last":
        demos.sort(key=lambda row: row.fused_score)
    elif config.order_strategy == "nearest_first":
        demos.sort(key=lambda row: row.fused_score, reverse=True)
    elif config.order_strategy == "random":
        seed = config.random_seed + sum(row.candidate.sample_id for row in demos)
        random.Random(seed).shuffle(demos)
    return demos


def _select_tu_demonstrations_baseline(
    query: ApiSample,
    rows: list[RetrievalRow],
    top_k: int,
) -> list[RetrievalRow]:
    shortlist = [row for row in rows if row.candidate.source == "TU"]
    if not shortlist:
        shortlist = list(rows)

    selected: list[RetrievalRow] = []
    selected_ids: set[int] = set()

    def try_add(predicate) -> None:
        if len(selected) >= top_k:
            return
        for row in shortlist:
            if row.candidate.sample_id in selected_ids:
                continue
            if predicate(row):
                selected.append(row)
                selected_ids.add(row.candidate.sample_id)
                return

    for label in (1, 0):
        try_add(lambda row, label=label: row.candidate.api == query.api and row.candidate.label == label)
    for label in (1, 0):
        try_add(lambda row, label=label: row.candidate.label == label)
    while len(selected) < top_k:
        existing_apis = {row.candidate.api for row in selected}
        before = len(selected)
        try_add(lambda row, existing_apis=existing_apis: row.candidate.api not in existing_apis)
        if len(selected) >= top_k:
            break
        if len(selected) == before:
            break
        try_add(lambda row: True)
        if len(selected_ids) == len(shortlist):
            break

    return selected


def _select_tu_demonstrations(query: ApiSample, rows: list[RetrievalRow], top_k: int) -> list[RetrievalRow]:
    shortlist = [row for row in rows if row.candidate.source == "TU"]
    if not shortlist:
        shortlist = list(rows)

    query_features = build_sample_clue_features(query)
    if (
        query_features["likely_list_only"] == "yes"
        or query_features["likely_structural_only"] == "yes"
    ):
        target_pos = 1
    elif (
        query_features["api_focus"] in {"high", "medium"}
        and query_features["code_like"] == "yes"
    ):
        target_pos = min(3, max(1, top_k - 1))
    else:
        target_pos = min(2, max(1, top_k - 1))
    target_neg = max(1, top_k - target_pos)

    selected: list[RetrievalRow] = []
    selected_ids: set[int] = set()

    def try_add(predicate) -> None:
        if len(selected) >= top_k:
            return
        for row in shortlist:
            if row.candidate.sample_id in selected_ids:
                continue
            if predicate(row):
                selected.append(row)
                selected_ids.add(row.candidate.sample_id)
                return

    query_fragment = _normalized_fragment(query.fragment)
    # Exact same-KU examples are the strongest supervision for this task because
    # McGill labels relevance per target API, not per fragment.
    for label in (1, 0):
        try_add(
            lambda row, label=label: _normalized_fragment(row.candidate.fragment) == query_fragment
            and row.candidate.label == label
        )

    # First, anchor with same-API positive and same-API negative when possible.
    try_add(lambda row: row.candidate.api == query.api and row.candidate.label == 1)
    try_add(lambda row: row.candidate.api == query.api and row.candidate.label == 0)

    def count_label(label: int) -> int:
        return sum(1 for row in selected if row.candidate.label == label)

    # Fill positive/negative quotas with nearest same-source examples.
    while len(selected) < top_k and count_label(1) < target_pos:
        before = len(selected)
        try_add(lambda row: row.candidate.label == 1)
        if len(selected) == before or len(selected_ids) == len(shortlist):
            break
    while len(selected) < top_k and count_label(0) < target_neg:
        before = len(selected)
        try_add(lambda row: row.candidate.label == 0)
        if len(selected) == before or len(selected_ids) == len(shortlist):
            break

    while len(selected) < top_k:
        existing_apis = {row.candidate.api for row in selected}
        before = len(selected)
        try_add(lambda row, existing_apis=existing_apis: row.candidate.api not in existing_apis)
        if len(selected) >= top_k:
            break
        if len(selected) == before:
            break
        try_add(lambda row: True)
        if len(selected_ids) == len(shortlist):
            break

    return selected


def _select_so_demonstrations(query: ApiSample, rows: list[RetrievalRow], top_k: int) -> list[RetrievalRow]:
    shortlist = [row for row in rows if row.candidate.source == "SO"]
    if not shortlist:
        shortlist = list(rows)
    target_k = top_k

    selected: list[RetrievalRow] = []
    selected_ids: set[int] = set()

    def try_add(predicate) -> None:
        if len(selected) >= target_k:
            return
        for row in shortlist:
            if row.candidate.sample_id in selected_ids:
                continue
            if predicate(row):
                selected.append(row)
                selected_ids.add(row.candidate.sample_id)
                return

    def profile_of(row: RetrievalRow) -> str:
        return infer_sample_decision_profile(row.candidate)

    query_fragment = _normalized_fragment(query.fragment)
    for label in (1, 0):
        try_add(
            lambda row, label=label: _normalized_fragment(row.candidate.fragment) == query_fragment
            and row.candidate.label == label
        )

    # Positive: same API if possible.
    try_add(
        lambda row: row.candidate.api == query.api
        and row.candidate.label == 1
        and profile_of(row) in SO_POSITIVE_PROFILES
    )
    # Negative: same API but unresolved or clearly weak.
    try_add(
        lambda row: row.candidate.api == query.api
        and row.candidate.label == 0
        and profile_of(row) == "so_unresolved_question"
    )
    # Positive: solved answer from SO even if API differs, to anchor QA quality.
    try_add(
        lambda row: row.candidate.label == 1
        and profile_of(row) in SO_POSITIVE_PROFILES
    )
    # Negative: useful-looking answer but wrong API focus.
    try_add(
        lambda row: row.candidate.api != query.api
        and row.candidate.label == 0
        and profile_of(row) in {"so_wrong_api_focus", "so_solution_but_target_unclear"}
    )

    while len(selected) < target_k:
        existing_apis = {row.candidate.api for row in selected}
        try_add(lambda row, existing_apis=existing_apis: row.candidate.api not in existing_apis)
        if len(selected) >= target_k:
            break
        try_add(lambda row: True)
        if len(selected_ids) == len(shortlist):
            break

    return selected


def select_demonstrations(
    query: ApiSample,
    retrieved_rows: list[RetrievalRow],
    config: ICLConfig,
) -> list[RetrievalRow]:
    if query.source == "TU":
        return _select_tu_demonstrations(query, retrieved_rows, config.top_k)
    if query.source == "SO":
        return _select_so_demonstrations(query, retrieved_rows, config.top_k)
    return list(retrieved_rows[: config.top_k])


def _format_demonstrations(
    demos: list[RetrievalRow],
    config: ICLConfig,
    demo_augmentations: dict[int, DemoAugmentation] | None = None,
) -> list[str]:
    blocks: list[str] = []
    for idx, row in enumerate(_ordered_rows(demos, config), start=1):
        augmentation = (demo_augmentations or {}).get(row.candidate.sample_id) or fallback_demo_augmentation(row.candidate)
        label_text = _label_name(row.candidate.label)
        evidence = ""
        if config.evidence_augmented:
            evidence = (
                "\nRetrieval Evidence: "
                f"lexical_rank={row.lexical_rank}; semantic_rank={row.semantic_rank}; "
                f"structural_rank={row.structural_rank}; fused={row.fused_score:.4f}"
            )
        clue_block = ""
        if config.use_clue_enhancement:
            clue_block = f"Clues:\n{augmentation.clues}\nReasoning:\n{augmentation.reasoning}\n"
        blocks.append(
            (
                f"Demonstration {idx}\n"
                f"API: {row.candidate.api}\n"
                f"Knowledge Unit:\n{row.candidate.fragment[:2400]}\n"
                + clue_block
                + f"<LABEL>{label_text}</LABEL>{evidence}"
            )
        )
    return blocks


def build_tu_prompt(
    query: ApiSample,
    demonstrations: list[RetrievalRow],
    config: ICLConfig,
    demo_augmentations: dict[int, DemoAugmentation] | None = None,
) -> str:
    focus_fragment = build_focus_fragment(query)
    if _is_android_sample(query):
        blocks = [
            "Task: judge whether the Android tutorial fragment is relevant to the target API.",
            "Feature Clues are heuristic preprocessing signals derived from the raw fragment. Use them as supporting evidence, not as labels.",
            "Focused Fragment is a deterministic preprocessing view that keeps the most instructional span when list or structural noise is present.",
            "Positive means: the fragment helps an Android developer understand what the target API/component is for, when to use it, what workflow or app behavior it supports, or what helper/base class Android recommends around it.",
            "For Android docs, high-level component guidance can still be positive even without a full step-by-step recipe.",
            "A fragment can be positive if it explains the API's role, lifecycle contract, persistence/backup behavior, recommended subclass/helper, or setup path in a way that helps someone decide how to use it in an app.",
            "Do not require direct method-by-method instruction for Android tutorial positives.",
            "Return label 0 when the target API is only a cross-reference, neighboring helper, package list item, or example mention and the fragment does not actually explain that target API.",
            "Return label 0 when the fragment is mainly generic platform overview text and the target API is not one of the main explained APIs.",
            "If the target is a package/topic overview page, it can still be positive when the fragment genuinely explains that package/topic and how it is used in Android development.",
            "Do not use StackOverflow answer-quality logic here. Judge only tutorial-style API usefulness.",
            "Output JSON only.",
        ]
    else:
        blocks = [
            "Task: judge whether the tutorial fragment is relevant to the target API.",
            "Feature Clues are heuristic preprocessing signals derived from the raw fragment. Use them as supporting evidence, not as labels.",
            "Focused Fragment is a deterministic preprocessing view that keeps the most instructional span when list or structural noise is present.",
            "Positive means: the fragment helps an unfamiliar reader understand when or how to use the target API for a programming task.",
            "The target API must be a main explained API in the fragment. Do not mark it relevant only because the fragment explains a broader concept, a containing class, a related implementation, or a neighboring API.",
            "A fragment can still be positive when the target API appears together with helper APIs, as long as the target API is materially used in the explained workflow or solution path.",
            "Do not require full method-by-method instruction. Concise but practical role/usage guidance can be positive.",
            "A conceptual explanation can be positive when it clearly explains the target API's role, behavior, contract, or semantics in a way that helps usage decisions.",
            "Do not treat tutorial_structural_like as automatic negative. Use it together with api_focus and whether the text gives actionable or decision-useful understanding.",
            "If the fragment centers on defining the target API concept (or when to choose it against nearby types), that can be label 1 even without step-by-step code.",
            "Return label 0 when the API is listed only for completeness, used only for comparison, shown as a mere example, appears as one implementation among several, or appears only in structural/Javadoc-like facts.",
            "Return label 0 when the fragment only shows static relationships, gives too little information to be meaningful, or you are unsure.",
            "If likely_list_only=yes and there is no substantive explanation of the target API, return 0.",
            "Do not use StackOverflow answer-quality logic here. Judge only tutorial-style API usefulness.",
            "In reason, cite 1-2 decisive clues (for example: api_focus=high, likely_list_only=yes).",
            "Output JSON only.",
        ]
    blocks.extend(_format_demonstrations(demonstrations, config, demo_augmentations))
    blocks.append(
        (
            "Query\n"
            f"Source: {_source_name(query.source)}\n"
            f"Dataset: {query.dataset}\n"
            f"Target API: {query.api}\n"
            f"Feature Clues: {render_sample_clue_text(query)}\n"
            + (f"Focused Fragment:\n{focus_fragment}\n" if focus_fragment and focus_fragment != query.fragment[:1600] else "")
            + f"Fragment:\n{query.fragment[:2800]}\n"
            + 'Respond as JSON, for example {"label": 1, "reason": "..."}.' 
        )
    )
    return "\n\n".join(blocks)


def build_so_prompt(
    query: ApiSample,
    demonstrations: list[RetrievalRow],
    config: ICLConfig,
    demo_augmentations: dict[int, DemoAugmentation] | None = None,
) -> str:
    focus_fragment = build_focus_fragment(query)
    if _is_android_sample(query):
        blocks = [
            "Task: judge whether the StackOverflow fragment is relevant to the target API.",
            "Treat the fragment as a mixed Q/A artifact: it may contain the question, follow-up edits, and one or more answers together.",
            "Read the Focused Fragment first. It is a deterministic preprocessing view that tries to suppress question noise and keep the most decision-relevant answer span.",
            "Return label 1 only when both conditions hold:",
            "1. The fragment contains a useful answer, workaround, explanation, or reusable fix that solves or materially advances the problem.",
            "2. The target API is part of the solved QA context and is not merely incidental.",
            "For Android QA, several framework APIs may appear together. The target API can still be positive when the answer clearly shows how it participates in the reusable fix or framework workflow.",
            "Return label 0 if the answer is useful but the target API is only background to another API, only a surrounding container/signature, or only incidental setup.",
            "Do not apply tutorial-style standards here. A StackOverflow positive can be short, fix-oriented, and centered on solving the concrete QA.",
            "It does not need to teach the API like a tutorial. A correct, reusable fix is enough.",
            "If the fragment contains both a question and a solution, judge the solution content rather than punishing the sample for starting with a question.",
            "A concise but complete code recipe or corrected snippet counts as positive if it genuinely solves the API-related problem.",
            "Do not reject a fragment just because it starts with a question. Many StackOverflow positives contain both the question and the answer; judge the useful answer content.",
            "If the answer quality is weak or the target API is truly incidental, return 0.",
            "If unsure, return 0.",
            "Output JSON only.",
        ]
    else:
        blocks = [
            "Task: judge whether the StackOverflow fragment is relevant to the target API.",
            "Treat the fragment as a mixed Q/A artifact: it may contain the question, follow-up edits, and one or more answers together.",
            "Feature Clues are heuristic signals. Use them to structure judgment, not as hard labels.",
            "Read the Focused Fragment first. It is a deterministic preprocessing view that tries to suppress question noise and keep the most decision-relevant answer span.",
            "The target API must be part of the answer's reusable solution or explanation, not merely a class mentioned in the question, stack trace, type signature, or surrounding Joda-Time context.",
            "Return label 1 only when both conditions hold:",
            "1. The fragment contains a useful answer, workaround, explanation, or reusable fix that solves or materially advances the problem.",
            "2. The target API is part of the solved QA context and is not merely incidental.",
            "If solution_like=yes and answer_excerpt_api_mentions>=1 and likely_wrong_api_focus=no, that is strong positive evidence.",
            "Return label 0 if the fragment is mainly an unresolved question, symptom dump, weak answer, non-solution, or only general discussion.",
            "question_like=yes or unresolved_like=yes alone is not enough for 0 if the answer span still contains a concrete reusable fix.",
            "Return label 0 if the answer is useful but the target API is only background to another API, another nearby class, or a general programming issue.",
            "Return label 0 if the target API is only a carrier/argument/container while the decision-critical fix is actually about another API.",
            "Do not apply tutorial-style standards here. A StackOverflow positive can be short, fix-oriented, and centered on solving the concrete QA.",
            "If a nearby class appears in the fix, that can still be positive when the answer is clearly reusable for the target API problem.",
            "It does not need to teach the API like a tutorial. A correct, reusable fix is enough.",
            "Generic advice is not enough. The useful part must still be concretely reusable for this API-related QA.",
            "If the fragment contains both a question and a solution, judge the solution content rather than punishing the sample for starting with a question.",
            "A concise but complete code recipe or corrected snippet counts as positive if it genuinely solves the API-related problem.",
            "Do not reject a fragment just because it starts with a question. Many StackOverflow positives contain both the question and the answer; judge the useful answer content.",
            "If the answer quality is weak or the target API is truly incidental, return 0.",
            "If unsure, return 0.",
            "In reason, cite 1-2 decisive clues (for example: solution_like=yes, likely_wrong_api_focus=no).",
            "Output JSON only.",
        ]
    blocks.extend(_format_demonstrations(demonstrations, config, demo_augmentations))
    blocks.append(
        (
            "Query\n"
            f"Source: {_source_name(query.source)}\n"
            f"Dataset: {query.dataset}\n"
            f"Target API: {query.api}\n"
            f"Feature Clues: {render_sample_clue_text(query)}\n"
            + (f"Focused Fragment:\n{focus_fragment}\n" if focus_fragment and focus_fragment != query.fragment[:1600] else "")
            + f"Fragment:\n{query.fragment[:2800]}\n"
            'Respond as JSON, for example {"label": 1, "reason": "..."}.' 
        )
    )
    return "\n\n".join(blocks)


def build_lamic_prompt(
    query: ApiSample,
    demonstrations: list[RetrievalRow],
    config: ICLConfig,
    demo_augmentations: dict[int, DemoAugmentation] | None = None,
) -> str:
    blocks = [
        "Task Description: determine whether a knowledge unit (KU) is relevant to a target API.",
        "A KU is Relevant if it explains how the API is used, its role or behavior, constraints, usage scenario, or an API-related problem solution.",
        "A KU is Irrelevant if it only mentions the API superficially, focuses on another API, or does not provide meaningful usage knowledge for the target API.",
        "Use the demonstrations as in-context examples. The demonstrations were selected by lexical, semantic, and structural retrieval perspectives.",
    ]
    if config.use_clue_enhancement:
        blocks.extend(
            [
                "Each demonstration includes Clues and Reasoning before the label.",
                "For the query, first output <CLUES>...</CLUES>, then <REASONING>...</REASONING>, and finally <LABEL>Relevant</LABEL> or <LABEL>Irrelevant</LABEL>.",
            ]
        )
    else:
        blocks.extend(
            [
                "Let's think step by step before the final decision.",
                "For the query, output <REASONING>...</REASONING> and finally <LABEL>Relevant</LABEL> or <LABEL>Irrelevant</LABEL>.",
            ]
        )
    blocks.extend(_format_demonstrations(demonstrations, config, demo_augmentations))
    query_prefix = (
        "Query\n"
        f"API: {query.api}\n"
        f"Knowledge Unit:\n{query.fragment[:2800]}\n"
    )
    if config.use_clue_enhancement:
        query_prefix += "Respond with <CLUES>, <REASONING>, and <LABEL>."
    else:
        query_prefix += "Respond with <REASONING> and <LABEL>."
    blocks.append(query_prefix)
    return "\n\n".join(blocks)


def build_prompt(
    query: ApiSample,
    demonstrations: list[RetrievalRow],
    config: ICLConfig,
    demo_augmentations: dict[int, DemoAugmentation] | None = None,
) -> str:
    if query.source == "SO":
        return build_so_prompt(query, demonstrations, config, demo_augmentations)
    if query.source == "TU":
        return build_tu_prompt(query, demonstrations, config, demo_augmentations)
    return build_lamic_prompt(query, demonstrations, config, demo_augmentations)


def build_so_verification_prompt(
    query: ApiSample,
    demonstrations: list[RetrievalRow],
    stage1_reason: str,
    risk_flags: list[str],
    config: ICLConfig,
) -> str:
    focus_fragment = build_focus_fragment(query)
    clue_text = render_sample_clue_text(query)
    blocks: list[str] = [
        "Task: verify whether a current StackOverflow positive prediction should remain positive.",
        "This verifier is only for borderline cases where the first pass may have over-trusted question text or generic advice.",
        "Current predicted label: 1",
        f"Stage-1 reason: {stage1_reason or 'none'}",
        f"Risk flags: {', '.join(risk_flags) if risk_flags else 'none'}",
        "Keep label 1 only if the answer-focused content contains a concrete, reusable fix or explanation and the target API is genuinely part of that solution.",
        "Return label 0 if the fragment is still mainly a question, unresolved discussion, weak answer, generic date-time advice, or a solution centered on another API.",
        "Return label 0 if the target API is only incidental background, even when nearby Joda-Time classes appear in the answer.",
        "Use the Focused Fragment first, but do not demote solely because it is short or partial; consult the full Fragment when the focused view may have dropped useful context.",
        "A positive can be conceptual or fix-oriented guidance about the target API, not only a direct method call.",
        "If the full Fragment clearly contains a reusable answer span for the target API, keep label 1 even when the question text is prominent.",
        "Feature Clues are heuristic signals, not labels.",
        "Output JSON only.",
    ]
    blocks.extend(_format_demonstrations(demonstrations, config))
    blocks.append(
        (
            "Query\n"
            f"Source: {_source_name(query.source)}\n"
            f"Dataset: {query.dataset}\n"
            f"Target API: {query.api}\n"
            f"Feature Clues: {clue_text}\n"
            + (f"Focused Fragment:\n{focus_fragment}\n" if focus_fragment and focus_fragment != query.fragment[:1600] else "")
            + f"Fragment:\n{query.fragment[:2800]}\n"
            + 'Respond as JSON, for example {"label": 0, "reason": "..."}.' 
        )
    )
    return "\n\n".join(blocks)


def parse_prediction(raw_response: str) -> Prediction:
    label_match = re.search(r"<LABEL>\s*(Relevant|Irrelevant|1|0)\s*</LABEL>", raw_response, re.IGNORECASE)
    if label_match:
        raw_label = label_match.group(1).lower()
        label = 1 if raw_label in {"relevant", "1"} else 0
        reason_match = re.search(r"<REASONING>\s*(.*?)\s*</REASONING>", raw_response, re.IGNORECASE | re.DOTALL)
        return Prediction(
            label=label,
            reason=reason_match.group(1).strip() if reason_match else "",
            raw_response=raw_response,
            llm_label=label,
        )
    start = raw_response.find("{")
    end = raw_response.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model response is not JSON: {raw_response}")
    payload = json.loads(raw_response[start : end + 1])
    raw_label = payload["label"]
    if isinstance(raw_label, str) and raw_label.strip().lower() in {"relevant", "irrelevant"}:
        label = 1 if raw_label.strip().lower() == "relevant" else 0
    else:
        label = int(raw_label)
    if label not in {0, 1}:
        raise ValueError(f"Invalid label returned by model: {label}")
    return Prediction(
        label=label,
        reason=str(payload.get("reason", "")).strip(),
        raw_response=raw_response,
        llm_label=label,
    )


def calibrate_prediction(
    prediction: Prediction,
    query: ApiSample,
    demonstrations: list[RetrievalRow],
    config: ICLConfig,
) -> Prediction:
    if not config.enable_source_aware_calibration:
        return prediction

    if query.source != "SO" or not demonstrations:
        return prediction

    top_demo = demonstrations[0]
    positive_votes = sum(row.candidate.label == 1 for row in demonstrations)
    negative_votes = len(demonstrations) - positive_votes
    same_api_positive = any(
        row.candidate.api == query.api and row.candidate.label == 1 for row in demonstrations
    )

    adjusted_label = prediction.label
    required_positive_votes = max(3, (len(demonstrations) * 3 + 3) // 4)  # ceil(0.75 * k)

    if (
        prediction.label == 0
        and top_demo.candidate.label == 1
        and same_api_positive
        and positive_votes >= required_positive_votes
    ):
        adjusted_label = 1
    elif (
        prediction.label == 1
        and top_demo.candidate.label == 0
        and negative_votes == len(demonstrations)
        and not same_api_positive
    ):
        adjusted_label = 0

    if adjusted_label == prediction.label:
        return prediction

    return Prediction(
        label=adjusted_label,
        reason=prediction.reason,
        raw_response=prediction.raw_response,
        llm_label=prediction.llm_label,
        calibration_applied=True,
        calibration_note="SO source-aware calibration applied",
    )


def verify_so_positive_prediction(
    client: DeepSeekClient,
    query: ApiSample,
    demonstrations: list[RetrievalRow],
    prediction_row: dict,
    config: ICLConfig,
) -> dict:
    updated = dict(prediction_row)
    should_verify, risk_flags = should_verify_so_positive(updated, query)
    updated["so_verifier_applied"] = False
    updated["so_verifier_changed"] = False
    updated["so_verifier_risk_flags"] = risk_flags

    if not should_verify:
        return updated

    prompt = build_so_verification_prompt(
        query=query,
        demonstrations=demonstrations,
        stage1_reason=str(updated.get("reason", "")),
        risk_flags=risk_flags,
        config=config,
    )
    verifier_prediction = parse_prediction(client.chat(prompt))
    updated["so_verifier_applied"] = True
    updated["so_verifier_predicted_label"] = verifier_prediction.label
    updated["so_verifier_reason"] = verifier_prediction.reason
    updated["so_verifier_raw_response"] = verifier_prediction.raw_response

    if verifier_prediction.label == 0 and int(updated["predicted_label"]) == 1:
        updated["stage1_reason"] = str(updated.get("reason", ""))
        updated["predicted_label"] = 0
        updated["reason"] = verifier_prediction.reason
        updated["so_verifier_changed"] = True

    return updated
