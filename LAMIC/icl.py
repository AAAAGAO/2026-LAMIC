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
from .clues import build_sample_clue_features, extract_answer_focused_excerpt, infer_sample_decision_profile, render_demo_reason, render_sample_clue_text
from .config import ICLConfig
from .data import ApiSample
from .retrieval import RetrievalRow
SYSTEM_PROMPT = 'You are an API knowledge identification assistant.\nDetermine whether the given knowledge unit is relevant to the target API.\nRelevant means the knowledge unit explains how the API is used, what role it plays, or how it helps solve an API-related task.\nReturn the final decision inside <LABEL> and </LABEL> as Relevant or Irrelevant.'
ENHANCEMENT_SYSTEM_PROMPT = 'You generate clue-augmented demonstrations for API knowledge location.\nGiven a labeled <API, KU> example, output JSON with keys clues and reasoning.\nclues must summarize local evidence that supports the gold label.\nreasoning must briefly explain why the clues support that label.'
SO_POSITIVE_PROFILES = {'so_solution_with_target_api', 'so_concise_api_recipe'}
SO_NEGATIVE_PROFILES = {'so_unresolved_question', 'so_wrong_api_focus', 'so_solution_but_target_unclear', 'so_mixed_or_weak_qa'}

def _is_android_sample(sample: ApiSample) -> bool:
    return sample.language == 'android'

@dataclass(slots=True)
class Prediction:
    label: int
    reason: str
    raw_response: str
    llm_label: int

@dataclass(slots=True)
class DemoAugmentation:
    clues: str
    reasoning: str
    raw_response: str = ''
    generated_by_llm: bool = False

class DeepSeekClient:

    def __init__(self, config: ICLConfig) -> None:
        self.api_key = config.api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError('DeepSeek API key is required. Pass --api-key or set DEEPSEEK_API_KEY.')
        self.model_name = config.model_name
        self.url = config.url
        self.timeout_seconds = config.timeout_seconds
        self.max_retries = config.max_retries
        self.headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}

    def chat(self, prompt: str, system_prompt: str=SYSTEM_PROMPT) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(self.url, headers=self.headers, json={'model': self.model_name, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}], 'temperature': 0.0, 'max_tokens': 1024, 'top_p': 1.0, 'presence_penalty': 0.0, 'frequency_penalty': 0.0}, timeout=self.timeout_seconds)
                response.raise_for_status()
                data = response.json()
                content = str(data['choices'][0]['message']['content'])
                if not content.strip():
                    raise RuntimeError('DeepSeek returned an empty response')
                return content
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f'DeepSeek request failed after {self.max_retries} attempts') from last_error

def fallback_demo_augmentation(sample: ApiSample) -> DemoAugmentation:
    return DemoAugmentation(clues=render_sample_clue_text(sample), reasoning=render_demo_reason(sample), generated_by_llm=False)

def _label_name(label: int) -> str:
    return 'Relevant' if int(label) == 1 else 'Irrelevant'

def build_demo_enhancement_prompt(sample: ApiSample) -> str:
    return '\n\n'.join(['Create clue-augmented reasoning for this labeled demonstration.', 'Focus on whether the target API itself is the main explained or solved API, not merely a nearby class, return type, example value, or list item.', 'For StackOverflow, separate the question text from the answer/fix. A useful answer for another API should not make the target API relevant.', 'For tutorials, distinguish substantive usage/role explanation from API lists, implementation inventories, and Javadoc-like structural facts.', f'API: {sample.api}', f'Gold Label: {_label_name(sample.label)}', f'Knowledge Unit:\n{sample.fragment[:2800]}', 'Output JSON only with keys clues and reasoning. clues should name decisive evidence such as target_api_focus, answer_or_tutorial_focus, usage_or_role_evidence, and incidental_or_list_evidence.'])

def parse_demo_augmentation(raw_response: str) -> DemoAugmentation:
    start = raw_response.find('{')
    end = raw_response.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f'Enhancement response is not JSON: {raw_response}')
    payload = json.loads(raw_response[start:end + 1])
    raw_clues = payload.get('clues', '')
    if isinstance(raw_clues, list):
        clues = '; '.join((str(item).strip() for item in raw_clues if str(item).strip()))
    else:
        clues = str(raw_clues).strip()
    reasoning = str(payload.get('reasoning', '')).strip()
    if not clues or not reasoning:
        raise ValueError(f'Enhancement response misses clues or reasoning: {raw_response}')
    return DemoAugmentation(clues=clues, reasoning=reasoning, raw_response=raw_response, generated_by_llm=True)

class ClueBasedDemoEnhancer:

    def __init__(self, client: DeepSeekClient, cache_path: str | Path | None=None) -> None:
        self.client = client
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict[str, str | bool]] = {}
        self._lock = threading.Lock()
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                self.cache = {}

    def _cache_key(self, sample: ApiSample) -> str:
        return f'{sample.sample_id}:{sample.label}'

    def enhance(self, sample: ApiSample) -> DemoAugmentation:
        key = self._cache_key(sample)
        with self._lock:
            cached = self.cache.get(key)
        if cached:
            return DemoAugmentation(clues=str(cached.get('clues', '')), reasoning=str(cached.get('reasoning', '')), raw_response=str(cached.get('raw_response', '')), generated_by_llm=bool(cached.get('generated_by_llm', True)))
        try:
            augmentation = parse_demo_augmentation(self.client.chat(build_demo_enhancement_prompt(sample), system_prompt=ENHANCEMENT_SYSTEM_PROMPT))
        except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
            fallback = fallback_demo_augmentation(sample)
            augmentation = DemoAugmentation(clues=fallback.clues, reasoning=f'{fallback.reasoning} LLM clue enhancement failed: {exc}', raw_response='', generated_by_llm=False)
        with self._lock:
            self.cache[key] = {'clues': augmentation.clues, 'reasoning': augmentation.reasoning, 'raw_response': augmentation.raw_response, 'generated_by_llm': augmentation.generated_by_llm}
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
                with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False, dir=self.cache_path.parent, prefix=f'{self.cache_path.stem}.', suffix='.tmp') as handle:
                    handle.write(serialized)
                    temp_path = Path(handle.name)
                temp_path.replace(self.cache_path)
                return
            except OSError as exc:
                last_error = exc
                try:
                    if 'temp_path' in locals() and temp_path.exists():
                        temp_path.unlink()
                except OSError:
                    pass
                time.sleep(0.5 * (attempt + 1))
        raise OSError(f'Failed to save demo enhancement cache: {self.cache_path}') from last_error

def _source_name(source: str) -> str:
    return {'SO': 'StackOverflow', 'TU': 'Tutorial'}.get(source, source)

def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any((pattern in lowered for pattern in patterns))

def _normalized_fragment(text: str) -> str:
    return re.sub('\\s+', ' ', text).strip().lower()

def _is_question_like(fragment: str) -> bool:
    return _contains_any(fragment, ('?', 'how do i', 'how can i', 'any ideas', 'what should i use', 'why am i', 'i have this problem', 'it throws', 'exception', 'error', 'help'))

def _is_solution_like(fragment: str) -> bool:
    return _contains_any(fragment, ('try the following', 'you can use', 'the solution', 'workaround', 'use instead', 'for example', "here's", 'this works', 'returns', 'correct way', 'accepted answer'))

def _is_list_like_tutorial(fragment: str) -> bool:
    return _contains_any(fragment, ('complete list', 'implementations are', 'other implementations include', 'for instance', 'is represented by', 'the following values'))

def _ordered_rows(retrieved_rows: list[RetrievalRow], config: ICLConfig) -> list[RetrievalRow]:
    demos = list(retrieved_rows)
    if config.order_strategy == 'nearest_last':
        demos.sort(key=lambda row: row.fused_score)
    elif config.order_strategy == 'nearest_first':
        demos.sort(key=lambda row: row.fused_score, reverse=True)
    elif config.order_strategy == 'random':
        seed = config.random_seed + sum((row.candidate.sample_id for row in demos))
        random.Random(seed).shuffle(demos)
    return demos

def _select_tu_demonstrations_baseline(query: ApiSample, rows: list[RetrievalRow], top_k: int) -> list[RetrievalRow]:
    shortlist = [row for row in rows if row.candidate.source == 'TU']
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
    shortlist = [row for row in rows if row.candidate.source == 'TU']
    if not shortlist:
        shortlist = list(rows)
    query_features = build_sample_clue_features(query)
    if query_features['likely_list_only'] == 'yes' or query_features['likely_structural_only'] == 'yes':
        target_pos = 1
    elif query_features['api_focus'] in {'high', 'medium'} and query_features['code_like'] == 'yes':
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
    for label in (1, 0):
        try_add(lambda row, label=label: _normalized_fragment(row.candidate.fragment) == query_fragment and row.candidate.label == label)
    try_add(lambda row: row.candidate.api == query.api and row.candidate.label == 1)
    try_add(lambda row: row.candidate.api == query.api and row.candidate.label == 0)

    def count_label(label: int) -> int:
        return sum((1 for row in selected if row.candidate.label == label))
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
    shortlist = [row for row in rows if row.candidate.source == 'SO']
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
        try_add(lambda row, label=label: _normalized_fragment(row.candidate.fragment) == query_fragment and row.candidate.label == label)
    try_add(lambda row: row.candidate.api == query.api and row.candidate.label == 1 and (profile_of(row) in SO_POSITIVE_PROFILES))
    try_add(lambda row: row.candidate.api == query.api and row.candidate.label == 0 and (profile_of(row) == 'so_unresolved_question'))
    try_add(lambda row: row.candidate.label == 1 and profile_of(row) in SO_POSITIVE_PROFILES)
    try_add(lambda row: row.candidate.api != query.api and row.candidate.label == 0 and (profile_of(row) in {'so_wrong_api_focus', 'so_solution_but_target_unclear'}))
    while len(selected) < target_k:
        existing_apis = {row.candidate.api for row in selected}
        try_add(lambda row, existing_apis=existing_apis: row.candidate.api not in existing_apis)
        if len(selected) >= target_k:
            break
        try_add(lambda row: True)
        if len(selected_ids) == len(shortlist):
            break
    return selected

def select_demonstrations(query: ApiSample, retrieved_rows: list[RetrievalRow], config: ICLConfig) -> list[RetrievalRow]:
    if query.source == 'TU':
        return _select_tu_demonstrations(query, retrieved_rows, config.top_k)
    if query.source == 'SO':
        return _select_so_demonstrations(query, retrieved_rows, config.top_k)
    return list(retrieved_rows[:config.top_k])

def _format_demonstrations(demos: list[RetrievalRow], config: ICLConfig, demo_augmentations: dict[int, DemoAugmentation] | None=None) -> list[str]:
    blocks: list[str] = []
    for idx, row in enumerate(_ordered_rows(demos, config), start=1):
        augmentation = (demo_augmentations or {}).get(row.candidate.sample_id) or fallback_demo_augmentation(row.candidate)
        label_text = _label_name(row.candidate.label)
        evidence = ''
        if config.evidence_augmented:
            evidence = f'\nRetrieval Evidence: lexical_rank={row.lexical_rank}; semantic_rank={row.semantic_rank}; structural_rank={row.structural_rank}; fused={row.fused_score:.4f}'
        clue_block = ''
        if config.use_clue_enhancement:
            clue_block = f'Clues:\n{augmentation.clues}\nReasoning:\n{augmentation.reasoning}\n'
        blocks.append(f'Demonstration {idx}\nAPI: {row.candidate.api}\nKnowledge Unit:\n{row.candidate.fragment[:2400]}\n' + clue_block + f'<LABEL>{label_text}</LABEL>{evidence}')
    return blocks

def build_lamic_prompt(query: ApiSample, demonstrations: list[RetrievalRow], config: ICLConfig, demo_augmentations: dict[int, DemoAugmentation] | None=None) -> str:
    blocks = ['Task Description: determine whether a knowledge unit (KU) is relevant to a target API.', 'A KU is Relevant if it explains how the API is used, its role or behavior, constraints, usage scenario, or an API-related problem solution.', 'A KU is Irrelevant if it only mentions the API superficially, focuses on another API, or does not provide meaningful usage knowledge for the target API.', 'Use the demonstrations as in-context examples. The demonstrations were selected by lexical, semantic, and structural retrieval perspectives.']
    if config.use_clue_enhancement:
        blocks.extend(['Each demonstration includes Clues and Reasoning before the label.', 'For the query, first output <CLUES>...</CLUES>, then <REASONING>...</REASONING>, and finally <LABEL>Relevant</LABEL> or <LABEL>Irrelevant</LABEL>.'])
    else:
        blocks.extend(["Let's think step by step before the final decision.", 'For the query, output <REASONING>...</REASONING> and finally <LABEL>Relevant</LABEL> or <LABEL>Irrelevant</LABEL>.'])
    blocks.extend(_format_demonstrations(demonstrations, config, demo_augmentations))
    query_prefix = f'Query\nAPI: {query.api}\nKnowledge Unit:\n{query.fragment[:2800]}\n'
    if config.use_clue_enhancement:
        query_prefix += 'Respond with <CLUES>, <REASONING>, and <LABEL>.'
    else:
        query_prefix += 'Respond with <REASONING> and <LABEL>.'
    blocks.append(query_prefix)
    return '\n\n'.join(blocks)

def build_prompt(query: ApiSample, demonstrations: list[RetrievalRow], config: ICLConfig, demo_augmentations: dict[int, DemoAugmentation] | None=None) -> str:
    return build_lamic_prompt(query, demonstrations, config, demo_augmentations)

def parse_prediction(raw_response: str) -> Prediction:
    label_match = re.search('<LABEL>\\s*(Relevant|Irrelevant|1|0)\\s*</LABEL>', raw_response, re.IGNORECASE)
    if label_match:
        raw_label = label_match.group(1).lower()
        label = 1 if raw_label in {'relevant', '1'} else 0
        reason_match = re.search('<REASONING>\\s*(.*?)\\s*</REASONING>', raw_response, re.IGNORECASE | re.DOTALL)
        return Prediction(label=label, reason=reason_match.group(1).strip() if reason_match else '', raw_response=raw_response, llm_label=label)
    start = raw_response.find('{')
    end = raw_response.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f'Model response is not JSON: {raw_response}')
    payload = json.loads(raw_response[start:end + 1])
    raw_label = payload['label']
    if isinstance(raw_label, str) and raw_label.strip().lower() in {'relevant', 'irrelevant'}:
        label = 1 if raw_label.strip().lower() == 'relevant' else 0
    else:
        label = int(raw_label)
    if label not in {0, 1}:
        raise ValueError(f'Invalid label returned by model: {label}')
    return Prediction(label=label, reason=str(payload.get('reason', '')).strip(), raw_response=raw_response, llm_label=label)
