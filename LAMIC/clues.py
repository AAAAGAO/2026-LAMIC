from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .data import ApiSample
from .preprocessing import split_api_tokens


QUESTION_PATTERNS = (
    "?",
    "how do i",
    "how can i",
    "any ideas",
    "what should i use",
    "why am i",
    "i have this problem",
    "please help",
    "can anyone explain",
    "what is wrong",
)
UNRESOLVED_PATTERNS = (
    "exception",
    "error",
    "invalid format",
    "doesn't work",
    "not working",
    "problem",
    "issue",
    "fails",
    "broken",
    "stacktrace",
)
SOLUTION_PATTERNS = (
    "try the following",
    "you can use",
    "use instead",
    "the solution",
    "workaround",
    "for example",
    "here's",
    "here is",
    "this works",
    "correct way",
    "accepted answer",
    "should be",
    "simply use",
    "i would suggest",
    "there is no good reason",
    "recommended",
    "you could",
    "just get",
    "the better option",
    "write",
)
TUTORIAL_LIST_PATTERNS = (
    "complete list",
    "implementations include",
    "other implementations include",
    "for instance",
    "the following values",
    "such as",
)
TUTORIAL_STRUCTURAL_PATTERNS = (
    "represented by",
    "base class",
    "interface",
    "returns a",
    "implementation of",
    "associated with",
)
CODE_MARKERS = ("```", "    ", "public ", "private ", "new ", ".")
ACCEPTED_ANSWER_PATTERNS = (
    "accepted answer",
    "correct answer",
    "other answers correctly",
)
FINAL_SOLUTION_PATTERNS = (
    "final solution",
    "the solution we finally went with",
    "here is an improvement",
    "here's solution",
    "here is the solution",
    "this works",
)
UNSOLVED_SELF_REPORT_PATTERNS = (
    "none of the answers helped",
    "still does not work",
    "didn't solve",
    "not solved",
    "i finally solved",
    "i solved this",
)
CORRECTION_PATTERNS = (
    "accepted answer is wrong",
    "answer is wrong",
    "wrong, but",
    "but only occasionally",
)
FEEDBACK_RULE_DESCRIPTIONS = {
    "so_unresolved_or_help_request": "SO negative: unresolved question, bug report, or help request without a reusable fix.",
    "so_target_api_incidental": "SO negative: useful answer exists, but the target API is only background to another solution.",
    "so_generic_datetime_advice": "SO negative: answer gives generic date-time advice rather than target-API-specific help.",
    "so_question_with_reusable_fix": "SO positive: question text is present, but the answer portion contains a reusable fix.",
    "so_answer_quality_signal": "SO positive: accepted-answer or final-solution wording suggests a solved QA artifact.",
    "tu_list_or_example_only": "TU negative: API is listed or used illustratively rather than taught.",
    "tu_structural_fact_only": "TU negative: fragment gives structural or Javadoc-like facts instead of usage guidance.",
    "tu_concise_usage_guidance": "TU positive: short fragment still gives actionable usage guidance.",
}


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def _target_api_terms(api: str) -> list[str]:
    raw = api.replace("()", "").replace("::", ".").strip()
    terms = {raw.lower()}
    for token in split_api_tokens(api):
        lowered = token.lower().strip()
        if lowered and len(lowered) >= 3:
            terms.add(lowered)
    return sorted(terms, key=len, reverse=True)


def count_target_api_mentions(api: str, fragment: str) -> int:
    lowered = fragment.lower()
    count = 0
    for term in _target_api_terms(api):
        escaped = re.escape(term)
        count += len(re.findall(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", lowered))
    return count


def _paragraphs(fragment: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", fragment) if part.strip()]
    return parts or [fragment.strip()]


def _solution_paragraph_score(api: str, paragraph: str) -> int:
    lowered = paragraph.lower()
    score = 0
    score += count_target_api_mentions(api, paragraph) * 2
    if _contains_any(lowered, SOLUTION_PATTERNS):
        score += 4
    if _contains_any(lowered, QUESTION_PATTERNS):
        score -= 2
    if _contains_any(lowered, UNRESOLVED_PATTERNS):
        score -= 1
    if ">" in paragraph:
        score += 1
    if "return" in lowered or "throws" in lowered or "use " in lowered:
        score += 1
    return score


def extract_answer_focused_excerpt(sample: ApiSample, max_paragraphs: int = 2) -> str:
    if sample.source != "SO":
        return ""
    paragraphs = _paragraphs(sample.fragment)
    scored = []
    for idx, paragraph in enumerate(paragraphs):
        score = _solution_paragraph_score(sample.api, paragraph)
        # Prefer later paragraphs slightly because SO fragments usually place answers after the question.
        score += idx // 3
        scored.append((score, idx, paragraph))
    best = [item for item in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True) if item[0] > 0]
    if not best:
        return ""
    selected = sorted(best[:max_paragraphs], key=lambda item: item[1])
    return "\n\n".join(paragraph for _, _, paragraph in selected)[:1600]


def build_sample_clue_features(sample: ApiSample) -> dict[str, str | int]:
    fragment = sample.fragment
    api_mentions = count_target_api_mentions(sample.api, fragment)
    answer_excerpt = extract_answer_focused_excerpt(sample)
    answer_excerpt_mentions = count_target_api_mentions(sample.api, answer_excerpt) if answer_excerpt else 0
    answer_excerpt_present = bool(answer_excerpt.strip())
    question_like = _contains_any(fragment, QUESTION_PATTERNS)
    unresolved_like = _contains_any(fragment, UNRESOLVED_PATTERNS)
    solution_like = _contains_any(fragment, SOLUTION_PATTERNS)
    tutorial_list_like = _contains_any(fragment, TUTORIAL_LIST_PATTERNS)
    tutorial_structural_like = _contains_any(fragment, TUTORIAL_STRUCTURAL_PATTERNS)
    code_like = _contains_any(fragment, CODE_MARKERS)
    concise = len(fragment.strip()) <= 900
    accepted_answer_like = _contains_any(fragment, ACCEPTED_ANSWER_PATTERNS)
    final_solution_like = _contains_any(fragment, FINAL_SOLUTION_PATTERNS)
    self_report_unsolved = _contains_any(fragment, UNSOLVED_SELF_REPORT_PATTERNS)
    correction_like = _contains_any(fragment, CORRECTION_PATTERNS)

    if api_mentions >= 3:
        api_focus = "high"
    elif api_mentions >= 1:
        api_focus = "medium"
    else:
        api_focus = "low"

    likely_wrong_api_focus = (
        sample.source == "SO" and solution_like and api_focus == "low"
    )
    likely_unsolved_qa = sample.source == "SO" and (self_report_unsolved or (question_like and not solution_like))
    likely_concise_api_recipe = (
        sample.source == "SO"
        and concise
        and solution_like
        and answer_excerpt_present
        and answer_excerpt_mentions >= 1
    )
    likely_general_advice = (
        sample.source == "SO"
        and solution_like
        and answer_excerpt_present
        and answer_excerpt_mentions == 0
        and api_focus != "high"
    )
    likely_list_only = sample.source == "TU" and tutorial_list_like
    likely_structural_only = sample.source == "TU" and tutorial_structural_like and not code_like

    return {
        "source": sample.source,
        "api_mentions": api_mentions,
        "answer_excerpt_api_mentions": answer_excerpt_mentions,
        "answer_excerpt_present": "yes" if answer_excerpt_present else "no",
        "api_focus": api_focus,
        "question_like": "yes" if question_like else "no",
        "unresolved_like": "yes" if unresolved_like else "no",
        "solution_like": "yes" if solution_like else "no",
        "accepted_answer_like": "yes" if accepted_answer_like else "no",
        "final_solution_like": "yes" if final_solution_like else "no",
        "self_report_unsolved": "yes" if self_report_unsolved else "no",
        "correction_like": "yes" if correction_like else "no",
        "tutorial_list_like": "yes" if tutorial_list_like else "no",
        "tutorial_structural_like": "yes" if tutorial_structural_like else "no",
        "code_like": "yes" if code_like else "no",
        "concise": "yes" if concise else "no",
        "likely_wrong_api_focus": "yes" if likely_wrong_api_focus else "no",
        "likely_unsolved_qa": "yes" if likely_unsolved_qa else "no",
        "likely_concise_api_recipe": "yes" if likely_concise_api_recipe else "no",
        "likely_general_advice": "yes" if likely_general_advice else "no",
        "likely_list_only": "yes" if likely_list_only else "no",
        "likely_structural_only": "yes" if likely_structural_only else "no",
    }


def infer_sample_decision_profile(sample: ApiSample) -> str:
    features = build_sample_clue_features(sample)
    if sample.source == "SO":
        if features["likely_unsolved_qa"] == "yes":
            return "so_unresolved_question"
        if features["self_report_unsolved"] == "yes":
            return "so_unresolved_question"
        if features["accepted_answer_like"] == "yes" and int(features["answer_excerpt_api_mentions"]) >= 1:
            return "so_solution_with_target_api"
        if features["final_solution_like"] == "yes" and int(features["answer_excerpt_api_mentions"]) >= 1:
            return "so_solution_with_target_api"
        if features["likely_concise_api_recipe"] == "yes":
            return "so_concise_api_recipe"
        if features["likely_wrong_api_focus"] == "yes" or features["likely_general_advice"] == "yes":
            return "so_wrong_api_focus"
        if features["solution_like"] == "yes" and int(features["answer_excerpt_api_mentions"]) >= 1:
            return "so_solution_with_target_api"
        if features["solution_like"] == "yes":
            return "so_solution_but_target_unclear"
        return "so_mixed_or_weak_qa"

    if features["likely_list_only"] == "yes":
        return "tu_list_only"
    if features["likely_structural_only"] == "yes":
        return "tu_structural_only"
    if features["api_focus"] in {"medium", "high"} and features["code_like"] == "yes":
        return "tu_actionable_usage"
    if features["api_focus"] in {"medium", "high"}:
        return "tu_brief_api_guidance"
    return "tu_incidental_mention"


def render_demo_reason(sample: ApiSample) -> str:
    profile = infer_sample_decision_profile(sample)
    if sample.source == "SO":
        if sample.label == 1:
            if profile == "so_concise_api_recipe":
                return "Strong answer gives a concise, reusable fix for the target-API scenario."
            if profile == "so_solution_with_target_api":
                return "Strong answer materially resolves the QA and the target API is part of the solution context."
            return "Strong answer is reusable for the target-API scenario even if other nearby APIs also appear."
        if profile == "so_unresolved_question":
            return "Mostly question or symptom text; no reusable fix is shown."
        if profile in {"so_wrong_api_focus", "so_solution_but_target_unclear"}:
            return "There may be answer content, but the target API is only background to a different issue."
        return "Answer quality is weak, unclear, or not useful enough to count as a good QA solution."

    if sample.label == 1:
        if profile == "tu_actionable_usage":
            return "Fragment explains how to use the target API for a concrete programming task."
        return "Fragment gives actionable guidance that helps understand when or how to use the target API."
    if profile == "tu_list_only":
        return "Target API is listed for completeness or illustration, not taught."
    if profile == "tu_structural_only":
        return "Fragment gives structural or Javadoc-like facts rather than usage guidance."
    return "Target API is mentioned, but the fragment does not teach meaningful usage."


def infer_feedback_rule_hits(sample: ApiSample) -> list[str]:
    features = build_sample_clue_features(sample)
    hits: list[str] = []
    if sample.source == "SO":
        if features["likely_unsolved_qa"] == "yes" or features["self_report_unsolved"] == "yes":
            hits.append("so_unresolved_or_help_request")
        if features["likely_wrong_api_focus"] == "yes":
            hits.append("so_target_api_incidental")
        if features["likely_general_advice"] == "yes":
            hits.append("so_generic_datetime_advice")
        if features["solution_like"] == "yes" and int(features["answer_excerpt_api_mentions"]) >= 1:
            hits.append("so_question_with_reusable_fix")
        if features["accepted_answer_like"] == "yes" or features["final_solution_like"] == "yes":
            hits.append("so_answer_quality_signal")
        return hits

    if features["likely_list_only"] == "yes":
        hits.append("tu_list_or_example_only")
    if features["likely_structural_only"] == "yes":
        hits.append("tu_structural_fact_only")
    if features["concise"] == "yes" and features["api_focus"] in {"medium", "high"} and features["code_like"] == "yes":
        hits.append("tu_concise_usage_guidance")
    return hits


def render_feedback_rule_text(sample: ApiSample) -> str:
    hits = infer_feedback_rule_hits(sample)
    if not hits:
        return "none"
    return "; ".join(FEEDBACK_RULE_DESCRIPTIONS[rule_id] for rule_id in hits)


def render_sample_clue_text(sample: ApiSample) -> str:
    features = build_sample_clue_features(sample)
    ordered_keys = [
        "api_mentions",
        "answer_excerpt_api_mentions",
        "answer_excerpt_present",
        "api_focus",
        "question_like",
        "unresolved_like",
        "solution_like",
        "accepted_answer_like",
        "final_solution_like",
        "self_report_unsolved",
        "correction_like",
        "tutorial_list_like",
        "tutorial_structural_like",
        "code_like",
        "concise",
        "likely_wrong_api_focus",
        "likely_unsolved_qa",
        "likely_concise_api_recipe",
        "likely_general_advice",
        "likely_list_only",
        "likely_structural_only",
    ]
    return "; ".join(f"{key}={features[key]}" for key in ordered_keys)


def export_augmented_samples(samples: list[ApiSample], path: str | Path) -> None:
    rows = []
    for sample in samples:
        features = build_sample_clue_features(sample)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "fragment": sample.fragment,
                "api": sample.api,
                "relevance": sample.label,
                "source": sample.source,
                "library": sample.library,
                "language": sample.language,
                "dataset": sample.dataset,
                "clue_features": render_sample_clue_text(sample),
                "decision_profile": infer_sample_decision_profile(sample),
                "heuristic_rationale": render_demo_reason(sample),
                "feedback_rule_hits": "|".join(infer_feedback_rule_hits(sample)),
                "feedback_rule_text": render_feedback_rule_text(sample),
                "answer_focused_excerpt": extract_answer_focused_excerpt(sample),
                **features,
            }
        )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
