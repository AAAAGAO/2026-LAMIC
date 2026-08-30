from __future__ import annotations
import hashlib
import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from statistics import mean
import torch
from .clues import build_sample_clue_features, export_augmented_samples, infer_sample_decision_profile, render_demo_reason, render_sample_clue_text
from .config import AppConfig
from .data import ANDROID_DATASETS, ApiSample, JAVA_DATASETS, build_kfold_splits, group_samples_by_library, load_samples, stratified_holdout, stratified_split
from .evaluation import case_studies, classification_metrics, retrieval_metrics
from .icl import ClueBasedDemoEnhancer, DeepSeekClient, Prediction, SO_POSITIVE_PROFILES, build_prompt, fallback_demo_augmentation, parse_prediction, select_demonstrations
from .retrieval import MultiPerspectiveRetriever
from .utils import dump_json, ensure_dir

def _subset_config(config: AppConfig, name: str) -> AppConfig:
    subset = deepcopy(config)
    subset.output_dir = config.output_dir / name
    return subset

def _metric_average(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted({key for row in metric_rows for key, value in row.items() if isinstance(value, (int, float))})
    return {key: mean([float(row[key]) for row in metric_rows if key in row]) for key in keys}

def _safe_classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    if not labels:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    return classification_metrics(labels, predictions)

def _language_metrics_from_prediction_rows(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:

    def grouped_metrics(prediction_key: str) -> dict[str, dict[str, float]]:
        labels_by_group: dict[str, list[int]] = {'overall': []}
        predictions_by_group: dict[str, list[int]] = {'overall': []}
        for row in rows:
            label = int(row['gold_label'])
            prediction = int(row[prediction_key])
            language = 'java' if str(row['dataset']).split('_', 1)[-1] in JAVA_DATASETS else 'android'
            labels_by_group.setdefault(language, []).append(label)
            predictions_by_group.setdefault(language, []).append(prediction)
            labels_by_group['overall'].append(label)
            predictions_by_group['overall'].append(prediction)
        return {group: _safe_classification_metrics(labels_by_group[group], predictions_by_group[group]) for group in ('overall', 'java', 'android') if group in labels_by_group}
    return {'raw': grouped_metrics('llm_predicted_label'), 'final': grouped_metrics('predicted_label')}

def _validate_library(config: AppConfig, grouped: dict[str, list[ApiSample]]) -> None:
    if config.library and config.library not in grouped:
        raise ValueError(f"Unknown library: {config.library}. Available: {', '.join(grouped)}")

def _filter_samples_by_source(samples: list[ApiSample], source: str | None) -> list[ApiSample]:
    if source is None:
        return samples
    return [sample for sample in samples if sample.source == source]

def _write_prediction_cache(cache_path, prediction_cache: dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(prediction_cache, indent=2, ensure_ascii=False)
    last_error: OSError | None = None
    for attempt in range(3):
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False, dir=cache_path.parent, prefix=f'{cache_path.stem}.', suffix='.tmp') as handle:
                handle.write(serialized)
                temp_path = Path(handle.name)
            temp_path.replace(cache_path)
            return
        except OSError as exc:
            last_error = exc
            try:
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            time.sleep(0.5 * (attempt + 1))
    raise OSError(f'Failed to save prediction cache: {cache_path}') from last_error

def _fallback_prediction(query: ApiSample, demonstrations) -> Prediction:
    profile = infer_sample_decision_profile(query)
    features = build_sample_clue_features(query)
    positive_demos = sum((demo.candidate.label == 1 for demo in demonstrations))
    negative_demos = len(demonstrations) - positive_demos
    if query.source == 'SO':
        label = int(profile in SO_POSITIVE_PROFILES and features['unresolved_like'] == 'no' and (int(features['answer_excerpt_api_mentions']) >= 1))
    elif query.source == 'TU':
        label = int(profile in {'tu_actionable_usage', 'tu_brief_api_guidance'} and features['api_focus'] == 'high' and (features['likely_list_only'] == 'no') and (positive_demos >= negative_demos))
    else:
        label = int(positive_demos >= max(1, negative_demos))
    reason = f'fallback after repeated invalid model responses; profile={profile}, positive_demos={positive_demos}, negative_demos={negative_demos}, clues={render_sample_clue_text(query)}'
    raw_response = json.dumps({'label': label, 'reason': reason}, ensure_ascii=False)
    return Prediction(label=label, reason=reason, raw_response=raw_response, llm_label=label)

def _get_prediction_with_retry(client: DeepSeekClient, prompt: str, cache_key: str, prediction_cache: dict[str, str], prediction_cache_path, query: ApiSample, demonstrations, config: AppConfig, prediction_cache_lock: threading.Lock | None=None) -> Prediction:
    if prediction_cache_lock is None:
        raw_prediction = prediction_cache.get(cache_key)
    else:
        with prediction_cache_lock:
            raw_prediction = prediction_cache.get(cache_key)
    if isinstance(raw_prediction, str) and (not raw_prediction.strip()):
        if prediction_cache_lock is None:
            prediction_cache.pop(cache_key, None)
        else:
            with prediction_cache_lock:
                prediction_cache.pop(cache_key, None)
        raw_prediction = None
    parse_errors: list[str] = []
    for attempt in range(1, config.icl.max_retries + 1):
        if raw_prediction is None:
            try:
                raw_prediction = client.chat(prompt)
            except RuntimeError as exc:
                parse_errors.append(f'attempt {attempt}: {exc}')
                raw_prediction = None
                continue
            if prediction_cache_lock is None:
                prediction_cache[cache_key] = raw_prediction
                _write_prediction_cache(prediction_cache_path, prediction_cache)
            else:
                with prediction_cache_lock:
                    prediction_cache[cache_key] = raw_prediction
                    _write_prediction_cache(prediction_cache_path, prediction_cache)
        try:
            return parse_prediction(raw_prediction)
        except (ValueError, json.JSONDecodeError, KeyError) as exc:
            parse_errors.append(f'attempt {attempt}: {exc}')
            if prediction_cache_lock is None:
                prediction_cache.pop(cache_key, None)
            else:
                with prediction_cache_lock:
                    prediction_cache.pop(cache_key, None)
            raw_prediction = None
    fallback = _fallback_prediction(query, demonstrations)
    if prediction_cache_lock is None:
        prediction_cache[cache_key] = fallback.raw_response
        _write_prediction_cache(prediction_cache_path, prediction_cache)
    else:
        with prediction_cache_lock:
            prediction_cache[cache_key] = fallback.raw_response
            _write_prediction_cache(prediction_cache_path, prediction_cache)
    return fallback

def _set_perspectives(config: AppConfig, lexical: bool, semantic: bool, structural: bool) -> None:
    config.model.perspectives.lexical = lexical
    config.model.perspectives.semantic = semantic
    config.model.perspectives.structural = structural

def run_icl_fold(config: AppConfig, pool: list[ApiSample], queries: list[ApiSample], output_name: str, retriever: MultiPerspectiveRetriever | None=None) -> dict:
    ensure_dir(config.output_dir)
    prediction_cache_path = config.output_dir / f'{output_name}_prediction_cache.json'
    if prediction_cache_path.exists():
        try:
            prediction_cache = json.loads(prediction_cache_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            prediction_cache = {}
    else:
        prediction_cache = {}
    if config.icl.max_queries is not None:
        queries = queries[:config.icl.max_queries]
    if retriever is None:
        device = torch.device('cuda' if config.device == 'cuda' and torch.cuda.is_available() else 'cpu')
        retriever = MultiPerspectiveRetriever(config.model, config.experiment.batch_size, device)
        retriever.fit(pool)
    client = DeepSeekClient(config.icl)
    enhancer = ClueBasedDemoEnhancer(client, config.output_dir / 'demo_enhancements.json')
    full_demo_augmentations: dict[int, object] = {}
    if config.icl.use_clue_enhancement and config.icl.generate_demo_clues_with_llm and config.icl.precompute_demo_clues:
        llm_workers = max(1, int(config.icl.llm_workers))
        if llm_workers == 1:
            for idx, sample in enumerate(pool, start=1):
                full_demo_augmentations[sample.sample_id] = enhancer.enhance(sample)
                enhancer.save()
                if idx % 10 == 0 or idx == len(pool):
                    print(f'enhanced_demo_clues={idx}/{len(pool)}', flush=True)
        else:
            with ThreadPoolExecutor(max_workers=llm_workers) as executor:
                futures = {executor.submit(enhancer.enhance, sample): sample for sample in pool}
                for idx, future in enumerate(as_completed(futures), start=1):
                    sample = futures[future]
                    full_demo_augmentations[sample.sample_id] = future.result()
                    if idx % 10 == 0 or idx == len(pool):
                        enhancer.save()
                        print(f'enhanced_demo_clues={idx}/{len(pool)}', flush=True)
            enhancer.save()
    labels = []
    predictions = []
    rankings = []
    prediction_rows = []
    prompt_rows_by_query: list[list[dict[str, str | int | float]]] = []
    prediction_cache_lock = threading.Lock()

    def process_query(query_index: int, query: ApiSample):
        rows = retriever.retrieve(query=query, top_k=max(50, config.icl.top_k * 10))
        demonstrations = select_demonstrations(query, rows, config.icl)
        demo_augmentations = {}
        if config.icl.use_clue_enhancement:
            for demo in demonstrations:
                if config.icl.generate_demo_clues_with_llm:
                    demo_augmentations[demo.candidate.sample_id] = full_demo_augmentations.get(demo.candidate.sample_id) or enhancer.enhance(demo.candidate)
                else:
                    demo_augmentations[demo.candidate.sample_id] = fallback_demo_augmentation(demo.candidate)
        prompt = build_prompt(query, demonstrations, config.icl, demo_augmentations)
        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        prediction_cache_key = f'{config.icl.model_name}:{query.sample_id}:{prompt_hash}'
        parsed_prediction = _get_prediction_with_retry(client, prompt, prediction_cache_key, prediction_cache, prediction_cache_path, query, demonstrations, config, prediction_cache_lock)
        result = parsed_prediction
        prompt_demo_rows = [{'sample_id': demo.candidate.sample_id, 'api': demo.candidate.api, 'label': demo.candidate.label, 'source': demo.candidate.source, 'lexical_rank': demo.lexical_rank, 'semantic_rank': demo.semantic_rank, 'structural_rank': demo.structural_rank, 'fused_score': demo.fused_score, 'clues': demo_augmentations.get(demo.candidate.sample_id, fallback_demo_augmentation(demo.candidate)).clues, 'reasoning': demo_augmentations.get(demo.candidate.sample_id, fallback_demo_augmentation(demo.candidate)).reasoning, 'decision_profile': infer_sample_decision_profile(demo.candidate), 'heuristic_rationale': render_demo_reason(demo.candidate)} for demo in demonstrations]
        prediction_row = {'sample_id': query.sample_id, 'api': query.api, 'source': query.source, 'dataset': query.dataset, 'predicted_label': result.label, 'llm_predicted_label': result.llm_label, 'reason': result.reason, 'raw_response': result.raw_response, 'query_clue_features': render_sample_clue_text(query), 'query_decision_profile': infer_sample_decision_profile(query), 'prompt_demo_labels': [demo.candidate.label for demo in demonstrations], 'prompt_demo_apis': [demo.candidate.api for demo in demonstrations], 'prompt_demo_sources': [demo.candidate.source for demo in demonstrations], 'prompt_demo_lexical_ranks': [demo.lexical_rank for demo in demonstrations], 'prompt_demo_semantic_ranks': [demo.semantic_rank for demo in demonstrations], 'prompt_demo_structural_ranks': [demo.structural_rank for demo in demonstrations], 'prompt_demo_decision_profiles': [infer_sample_decision_profile(demo.candidate) for demo in demonstrations], 'prompt_demo_clues': [demo_augmentations.get(demo.candidate.sample_id, fallback_demo_augmentation(demo.candidate)).clues for demo in demonstrations], 'prompt_demo_reasoning': [demo_augmentations.get(demo.candidate.sample_id, fallback_demo_augmentation(demo.candidate)).reasoning for demo in demonstrations], 'prompt_demo_heuristic_rationales': [render_demo_reason(demo.candidate) for demo in demonstrations]}
        prediction_row['gold_label'] = query.label
        return (query_index, query.label, int(prediction_row['predicted_label']), rows[:10], prompt_demo_rows, prediction_row)
    if max(1, int(config.icl.llm_workers)) == 1:
        query_results = [process_query(query_index, query) for query_index, query in enumerate(queries)]
    else:
        query_results = [None] * len(queries)
        with ThreadPoolExecutor(max_workers=max(1, int(config.icl.llm_workers))) as executor:
            futures = {executor.submit(process_query, query_index, query): query_index for query_index, query in enumerate(queries)}
            for completed, future in enumerate(as_completed(futures), start=1):
                query_index, label, prediction, ranking, prompt_demo_rows, prediction_row = future.result()
                query_results[query_index] = (query_index, label, prediction, ranking, prompt_demo_rows, prediction_row)
                if completed % 10 == 0 or completed == len(queries):
                    print(f'predicted_queries={completed}/{len(queries)}', flush=True)
    for result_row in query_results:
        _, label, prediction, ranking, prompt_demo_rows, prediction_row = result_row
        labels.append(label)
        predictions.append(prediction)
        rankings.append(ranking)
        prompt_rows_by_query.append(prompt_demo_rows)
        prediction_rows.append(prediction_row)
    metrics = classification_metrics(labels, predictions)
    metrics.update({f'retrieval_{key}': value for key, value in retrieval_metrics(queries, rankings).items()})
    dump_json(metrics, config.output_dir / f'{output_name}_metrics.json')
    dump_json(prediction_rows, config.output_dir / f'{output_name}_predictions.json')
    dump_json(case_studies(queries, rankings, config.icl.max_case_studies), config.output_dir / f'{output_name}_cases.json')
    enhancer.save()
    return metrics

def run_rq1(config: AppConfig) -> dict:
    samples = _filter_samples_by_source(load_samples(config.data_dir), config.source)
    grouped = group_samples_by_library(samples)
    _validate_library(config, grouped)
    selected_libraries = [config.library] if config.library else list(grouped.keys())
    ensure_dir(config.output_dir)
    per_library: dict[str, dict] = {}
    library_metrics = []
    for library in selected_libraries:
        subset_config = _subset_config(config, library)
        samples = grouped[library]
        export_augmented_samples(samples, subset_config.output_dir / f'{library}_augmented_samples.csv')
        folds = build_kfold_splits(samples, config.split.n_splits, config.experiment.seed)
        if config.rq_max_folds is not None:
            folds = folds[:max(1, config.rq_max_folds)]
        fold_metrics = []
        for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
            pool = [samples[idx] for idx in train_idx]
            queries = [samples[idx] for idx in test_idx]
            fold_metrics.append(run_icl_fold(subset_config, pool, queries, f'rq1_fold_{fold_id}'))
        result = {'folds': fold_metrics, 'macro_average': _metric_average(fold_metrics)}
        dump_json(result, subset_config.output_dir / 'rq1_results.json')
        per_library[library] = result
        library_metrics.append({'library': library, **result['macro_average']})
    summary = {'rq': 'RQ1', 'libraries': selected_libraries, 'library_metrics': library_metrics, 'macro_average': _metric_average(library_metrics)}
    dump_json(summary, config.output_dir / 'rq1_summary.json')
    return {'per_library': per_library, 'summary': summary}

def run_rq2(config: AppConfig) -> dict:
    samples = _filter_samples_by_source(load_samples(config.data_dir), config.source)
    grouped = group_samples_by_library(samples)
    _validate_library(config, grouped)
    selected_libraries = [config.library] if config.library else list(grouped.keys())
    ensure_dir(config.output_dir)
    per_library: dict[str, dict] = {}
    for library in selected_libraries:
        subset_config = _subset_config(config, library)
        samples = grouped[library]
        train_samples, _, test_samples = stratified_split(samples, train_size=config.split.train_size, dev_size=config.split.dev_size, test_size=config.split.test_size, seed=config.experiment.seed)
        ablations = {}
        for name, lexical, semantic, structural in [('lex', True, False, False), ('sem', False, True, False), ('str', False, False, True), ('lex_sem', True, True, False), ('lex_str', True, False, True), ('sem_str', False, True, True), ('lex_sem_str', True, True, True)]:
            ablation_config = deepcopy(subset_config)
            _set_perspectives(ablation_config, lexical, semantic, structural)
            ablations[name] = run_icl_fold(ablation_config, train_samples, test_samples, f'rq2_{name}')
        cot_config = deepcopy(subset_config)
        cot_config.icl.use_clue_enhancement = False
        cot_config.icl.generate_demo_clues_with_llm = False
        ablations['cot'] = run_icl_fold(cot_config, train_samples, test_samples, 'rq2_cot')
        dump_json(ablations, subset_config.output_dir / 'rq2_results.json')
        per_library[library] = ablations
    summary = {'rq': 'RQ2', 'libraries': selected_libraries}
    dump_json(summary, config.output_dir / 'rq2_summary.json')
    return {'per_library': per_library, 'summary': summary}

def run_rq3(config: AppConfig) -> dict:
    samples = _filter_samples_by_source(load_samples(config.data_dir), config.source)
    grouped = group_samples_by_library(samples)
    _validate_library(config, grouped)
    ensure_dir(config.output_dir)
    selected_libraries = [config.library] if config.library else list(grouped.keys())
    selected_samples = [sample for library in selected_libraries for sample in grouped[library]]
    selected_library_set = set(selected_libraries)
    if config.library is None and JAVA_DATASETS.issubset(selected_library_set) and ANDROID_DATASETS.issubset(selected_library_set):
        java_samples = [sample for sample in selected_samples if sample.library in JAVA_DATASETS]
        android_samples = [sample for sample in selected_samples if sample.library in ANDROID_DATASETS]
        java_test_size = min(config.rq3_test_size // 2, len(java_samples) - 1)
        android_test_size = min(config.rq3_test_size - java_test_size, len(android_samples) - 1)
        java_pool, java_queries = stratified_holdout(java_samples, test_size=java_test_size, seed=config.experiment.seed)
        android_pool, android_queries = stratified_holdout(android_samples, test_size=android_test_size, seed=config.experiment.seed + 1)
        pool = java_pool + android_pool
        queries = java_queries + android_queries
    else:
        pool, queries = stratified_holdout(selected_samples, test_size=min(config.rq3_test_size, len(selected_samples) - 1), seed=config.experiment.seed)
    base_config = deepcopy(config)
    base_config.icl.precompute_demo_clues = False
    base_config.icl.max_queries = None
    device = torch.device('cuda' if base_config.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    shared_retriever = MultiPerspectiveRetriever(base_config.model, base_config.experiment.batch_size, device)
    shared_retriever.fit(pool)
    factor_results = {}
    platform_results = {}
    for top_k in [0, 1, 3, 5, 10]:
        factor_config = deepcopy(base_config)
        factor_config.icl.top_k = top_k
        factor_config.icl.order_strategy = 'nearest_last'
        factor_results[f'top_k_{top_k}'] = run_icl_fold(factor_config, pool, queries, f'rq3_top_k_{top_k}', retriever=shared_retriever)
        prediction_rows = json.loads((config.output_dir / f'rq3_top_k_{top_k}_predictions.json').read_text(encoding='utf-8'))
        platform_results[f'top_k_{top_k}'] = _language_metrics_from_prediction_rows(prediction_rows)
    if config.rq3_run_order:
        for order_strategy in ['random', 'nearest_first']:
            factor_config = deepcopy(base_config)
            factor_config.icl.top_k = 5
            factor_config.icl.order_strategy = order_strategy
            factor_results[f'order_{order_strategy}'] = run_icl_fold(factor_config, pool, queries, f'rq3_order_{order_strategy}', retriever=shared_retriever)
            prediction_rows = json.loads((config.output_dir / f'rq3_order_{order_strategy}_predictions.json').read_text(encoding='utf-8'))
            platform_results[f'order_{order_strategy}'] = _language_metrics_from_prediction_rows(prediction_rows)
        factor_results['order_nearest_last'] = dict(factor_results['top_k_5'])
        platform_results['order_nearest_last'] = dict(platform_results['top_k_5'])
    strata: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    for sample in queries:
        key = f'{sample.library}:{sample.source}:{sample.label}'
        strata[key] = strata.get(key, 0) + 1
        language_counts[sample.language] = language_counts.get(sample.language, 0) + 1
    sample_manifest = {'seed': config.experiment.seed, 'test_size': len(queries), 'pool_size': len(pool), 'libraries': selected_libraries, 'language_counts': dict(sorted(language_counts.items())), 'strata': dict(sorted(strata.items())), 'test_samples': [{'sample_id': sample.sample_id, 'library': sample.library, 'language': sample.language, 'source': sample.source, 'label': sample.label, 'api': sample.api} for sample in queries]}
    dump_json(sample_manifest, config.output_dir / 'rq3_sample_manifest.json')
    dump_json(factor_results, config.output_dir / 'rq3_results.json')
    dump_json(platform_results, config.output_dir / 'rq3_platform_results.json')
    summary = {'rq': 'RQ3', 'libraries': selected_libraries, 'test_size': len(queries), 'language_counts': dict(sorted(language_counts.items())), 'pool_size': len(pool), 'seed': config.experiment.seed, 'run_order': config.rq3_run_order, 'note': 'order_nearest_last reuses top_k_5 because the settings are identical' if config.rq3_run_order else 'sample-size factors only'}
    dump_json(summary, config.output_dir / 'rq3_summary.json')
    return {'results': factor_results, 'platform_results': platform_results, 'summary': summary}

def run_rq4(config: AppConfig) -> dict:
    samples = _filter_samples_by_source(load_samples(config.data_dir), config.source)
    grouped = group_samples_by_library(samples)
    ensure_dir(config.output_dir)
    _validate_library(config, grouped)
    if config.rq4_query_library or config.rq4_pool_library:
        if not config.rq4_query_library or not config.rq4_pool_library:
            raise ValueError('RQ4 requires both rq4_query_library and rq4_pool_library when either is provided')
        if config.rq4_query_library not in grouped:
            raise ValueError(f'Unknown rq4_query_library: {config.rq4_query_library}')
        if config.rq4_pool_library not in grouped:
            raise ValueError(f'Unknown rq4_pool_library: {config.rq4_pool_library}')
        query_library = config.rq4_query_library
        pool_library = config.rq4_pool_library
        queries = grouped[query_library]
        pool = grouped[pool_library]
        query_language = queries[0].language
        pool_language = pool[0].language
        setting = 'within_language' if query_language == pool_language else 'cross_language'
        metrics = run_icl_fold(config, pool, queries, f'rq4_query_{query_library}_pool_{pool_library}')
        result = {'rq': 'RQ4', 'setting': setting, 'query_library': query_library, 'pool_library': pool_library, 'query_language': query_language, 'pool_language': pool_language, 'num_queries': len(queries) if config.icl.max_queries is None else min(len(queries), config.icl.max_queries), 'num_pool_samples': len(pool), 'metrics': metrics}
        dump_json(result, config.output_dir / 'rq4_results.json')
        return result
    selected_libraries = [config.library] if config.library else list(grouped.keys())
    selected_library_set = set(selected_libraries)
    if config.library is None and JAVA_DATASETS.issubset(selected_library_set) and ANDROID_DATASETS.issubset(selected_library_set):
        selected_samples = [sample for library in selected_libraries for sample in grouped[library]]
        java_samples = [sample for sample in selected_samples if sample.library in JAVA_DATASETS]
        android_samples = [sample for sample in selected_samples if sample.library in ANDROID_DATASETS]
        java_test_size = min(config.rq4_test_size // 2, len(java_samples) - 1)
        android_test_size = min(config.rq4_test_size - java_test_size, len(android_samples) - 1)
        java_pool, java_queries = stratified_holdout(java_samples, test_size=java_test_size, seed=config.experiment.seed)
        android_pool, android_queries = stratified_holdout(android_samples, test_size=android_test_size, seed=config.experiment.seed + 1)
        base_config = deepcopy(config)
        base_config.icl.precompute_demo_clues = False
        base_config.icl.max_queries = None
        device = torch.device('cuda' if base_config.device == 'cuda' and torch.cuda.is_available() else 'cpu')
        java_retriever = MultiPerspectiveRetriever(base_config.model, base_config.experiment.batch_size, device)
        android_retriever = MultiPerspectiveRetriever(base_config.model, base_config.experiment.batch_size, device)
        java_retriever.fit(java_pool)
        android_retriever.fit(android_pool)
        run_specs = [('within_language', 'java', java_pool, java_queries, java_retriever), ('within_language', 'android', android_pool, android_queries, android_retriever), ('cross_language', 'java', android_pool, java_queries, android_retriever), ('cross_language', 'android', java_pool, android_queries, java_retriever)]
        rows = []
        prediction_rows_by_setting: dict[str, list[dict]] = {}
        for setting, query_language, pool, queries, retriever in run_specs:
            output_name = f'rq4_{setting}_{query_language}'
            metrics = run_icl_fold(base_config, pool, queries, output_name, retriever=retriever)
            prediction_rows = json.loads((config.output_dir / f'{output_name}_predictions.json').read_text(encoding='utf-8'))
            prediction_rows_by_setting.setdefault(setting, []).extend(prediction_rows)
            rows.append({'setting': setting, 'query_language': query_language, 'pool_language': 'java' if pool and pool[0].library in JAVA_DATASETS else 'android', 'num_queries': len(queries), 'num_pool_samples': len(pool), 'raw': _language_metrics_from_prediction_rows(prediction_rows)['raw'], 'final': _language_metrics_from_prediction_rows(prediction_rows)['final'], 'metrics': metrics})
        platform_results = {setting: _language_metrics_from_prediction_rows(prediction_rows) for setting, prediction_rows in prediction_rows_by_setting.items()}
        queries = java_queries + android_queries
        pool = java_pool + android_pool
        strata: dict[str, int] = {}
        language_counts: dict[str, int] = {}
        for sample in queries:
            key = f'{sample.library}:{sample.source}:{sample.label}'
            strata[key] = strata.get(key, 0) + 1
            language_counts[sample.language] = language_counts.get(sample.language, 0) + 1
        sample_manifest = {'seed': config.experiment.seed, 'test_size': len(queries), 'pool_size': len(pool), 'libraries': selected_libraries, 'language_counts': dict(sorted(language_counts.items())), 'strata': dict(sorted(strata.items())), 'test_samples': [{'sample_id': sample.sample_id, 'library': sample.library, 'language': sample.language, 'source': sample.source, 'label': sample.label, 'api': sample.api} for sample in queries]}
        result = {'rq': 'RQ4', 'libraries': selected_libraries, 'test_size': len(queries), 'language_counts': dict(sorted(language_counts.items())), 'pool_size': len(pool), 'seed': config.experiment.seed, 'runs': rows, 'platform_results': platform_results}
        dump_json(sample_manifest, config.output_dir / 'rq4_sample_manifest.json')
        dump_json(platform_results, config.output_dir / 'rq4_platform_results.json')
        dump_json(result, config.output_dir / 'rq4_results.json')
        return result
    rows = []
    for query_library in selected_libraries:
        queries = grouped[query_library]
        query_language = queries[0].language
        for setting, pool_libraries in [('within_language', [library for library, library_samples in grouped.items() if library != query_library and library_samples[0].language == query_language]), ('cross_language', [library for library, library_samples in grouped.items() if library_samples[0].language != query_language])]:
            pool = [sample for library in pool_libraries for sample in grouped[library]]
            if not pool:
                continue
            subset_config = _subset_config(config, f'{query_library}_{setting}')
            metrics = run_icl_fold(subset_config, pool, queries, f'rq4_{setting}_query_{query_library}')
            rows.append({'setting': setting, 'query_library': query_library, 'pool_libraries': pool_libraries, 'query_language': query_language, 'num_queries': len(queries) if config.icl.max_queries is None else min(len(queries), config.icl.max_queries), 'num_pool_samples': len(pool), **metrics})
    result = {'rq': 'RQ4', 'libraries': selected_libraries, 'runs': rows, 'within_language_average': _metric_average([row for row in rows if row['setting'] == 'within_language']), 'cross_language_average': _metric_average([row for row in rows if row['setting'] == 'cross_language'])}
    dump_json(result, config.output_dir / 'rq4_results.json')
    return result

def run_rq_experiment(config: AppConfig) -> dict:
    if not config.rq_id:
        raise ValueError("rq command requires rq_id in {'RQ1', 'RQ2', 'RQ3', 'RQ4'}")
    rq_id = config.rq_id.upper()
    if rq_id == 'RQ1':
        return run_rq1(config)
    if rq_id == 'RQ2':
        return run_rq2(config)
    if rq_id == 'RQ3':
        return run_rq3(config)
    if rq_id == 'RQ4':
        return run_rq4(config)
    raise ValueError(f'Unknown rq_id: {config.rq_id}')
