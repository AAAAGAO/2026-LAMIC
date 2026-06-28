from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from hark.data import load_samples, samples_to_frame


ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner=False)
def load_dataset_frame(data_dir: str) -> pd.DataFrame:
    return samples_to_frame(load_samples(data_dir))


def list_json_files(output_dir: Path, suffix: str) -> list[Path]:
    return sorted(output_dir.rglob(f"*{suffix}"))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def get_library_options(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["library"].unique().tolist())


def build_cli_command(
    command: str,
    data_dir: str,
    output_dir: str,
    device: str,
    batch_size: int,
    epochs: int,
    top_k: int,
    grad_acc: int,
    seed: int,
    model_name: str,
    library: str | None,
    trained_output_dir: str | None,
    rq_id: str | None,
    rq4_query_library: str | None,
    rq4_pool_library: str | None,
    max_queries: int,
    api_key: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "1.py"),
        command,
        "--data-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--top-k",
        str(top_k),
        "--grad-accumulation-steps",
        str(grad_acc),
        "--seed",
        str(seed),
        "--model-name",
        model_name,
    ]
    if library:
        cmd.extend(["--library", library])
    if trained_output_dir:
        cmd.extend(["--trained-output-dir", trained_output_dir])
    if rq_id:
        cmd.extend(["--rq-id", rq_id])
    if rq4_query_library:
        cmd.extend(["--rq4-query-library", rq4_query_library])
    if rq4_pool_library:
        cmd.extend(["--rq4-pool-library", rq4_pool_library])
    if max_queries > 0:
        cmd.extend(["--max-queries", str(max_queries)])
    if api_key:
        cmd.extend(["--api-key", api_key])
    return cmd


def render_library_pair_summary(frame: pd.DataFrame, library: str) -> None:
    library_frame = frame[frame["library"] == library].copy()
    tu_count = int((library_frame["source"] == "TU").sum())
    so_count = int((library_frame["source"] == "SO").sum())
    relevant_rate = float(library_frame["label"].mean()) if not library_frame.empty else 0.0
    language = library_frame["language"].iloc[0] if not library_frame.empty else "-"
    api_count = int(library_frame["api"].nunique()) if not library_frame.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Library", library)
    c2.metric("Language", language)
    c3.metric("TU 样本", tu_count)
    c4.metric("SO 样本", so_count)
    c5.metric("Relevant 占比", f"{relevant_rate:.2%}")
    st.caption(f"该对数据集包含 {len(library_frame)} 条样本，{api_count} 个 API。")


def render_dataset_tab(frame: pd.DataFrame) -> None:
    st.subheader("数据总览")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("样本总数", int(len(frame)))
    col2.metric("数据集数量", int(frame["dataset"].nunique()))
    col3.metric("API 数量", int(frame["api"].nunique()))
    col4.metric("Relevant 占比", f"{frame['label'].mean():.2%}")

    st.caption("当前后端逻辑是：训练检索器时合并 9 对数据集统一训练，RQ 阶段再按单个 library 对分别测试。")

    left, right = st.columns(2)
    with left:
        st.write("按 dataset 统计")
        dataset_stats = (
            frame.groupby("dataset")
            .agg(samples=("sample_id", "count"), relevant_rate=("label", "mean"))
            .reset_index()
        )
        st.dataframe(dataset_stats, use_container_width=True)
    with right:
        st.write("按 source / language 统计")
        overview = (
            frame.groupby(["source", "language"])
            .agg(samples=("sample_id", "count"), relevant_rate=("label", "mean"))
            .reset_index()
        )
        st.dataframe(overview, use_container_width=True)

    st.subheader("样本浏览")
    dataset_options = ["ALL"] + sorted(frame["dataset"].unique().tolist())
    source_options = ["ALL"] + sorted(frame["source"].unique().tolist())
    language_options = ["ALL"] + sorted(frame["language"].unique().tolist())

    f1, f2, f3, f4 = st.columns(4)
    selected_dataset = f1.selectbox("dataset", dataset_options)
    selected_source = f2.selectbox("source", source_options)
    selected_language = f3.selectbox("language", language_options)
    selected_label = f4.selectbox("label", ["ALL", 0, 1])

    filtered = frame.copy()
    if selected_dataset != "ALL":
        filtered = filtered[filtered["dataset"] == selected_dataset]
    if selected_source != "ALL":
        filtered = filtered[filtered["source"] == selected_source]
    if selected_language != "ALL":
        filtered = filtered[filtered["language"] == selected_language]
    if selected_label != "ALL":
        filtered = filtered[filtered["label"] == selected_label]

    st.dataframe(
        filtered[["sample_id", "dataset", "source", "language", "api", "label"]].head(200),
        use_container_width=True,
        height=360,
    )

    if not filtered.empty:
        sample_id = st.selectbox("选择一个样本查看详情", filtered["sample_id"].tolist())
        row = filtered[filtered["sample_id"] == sample_id].iloc[0]
        st.markdown(f"**API**: `{row['api']}`")
        st.markdown(f"**Gold Label**: `{row['label']}`")
        st.markdown(f"**Dataset**: `{row['dataset']}` / `{row['source']}` / `{row['language']}`")
        st.text_area("Fragment", value=row["fragment"], height=320)


def render_runner_tab(frame: pd.DataFrame) -> None:
    st.subheader("测试任务发起")
    st.caption("直接在页面里选择动作和实验编号。这里仍然调用现有后端，但不需要你手写命令。")

    library_options = ["全部 9 对"] + get_library_options(frame)
    operation = st.radio(
        "执行内容",
        ["训练检索器", "运行 RQ 实验"],
        horizontal=True,
    )
    rq_id = None
    selected_library = "全部 9 对"
    rq4_query_library = None
    rq4_pool_library = None
    if operation == "运行 RQ 实验":
        rq_id = st.selectbox("RQ 编号", ["RQ1", "RQ2", "RQ3", "RQ4"])
        if rq_id in {"RQ1", "RQ2", "RQ3"}:
            selected_library = st.selectbox("选择测试数据对", library_options, index=0)
            if selected_library != "全部 9 对":
                render_library_pair_summary(frame, selected_library)
            else:
                st.info(f"{rq_id} 将对 9 个 library 对分别执行，不会在同一次测试里混合。")
        else:
            rq4_query_library = st.selectbox("RQ4 测试集 library", get_library_options(frame), index=0)
            rq4_pool_library = st.selectbox("RQ4 检索候选集 library", get_library_options(frame), index=1)
            render_library_pair_summary(frame, rq4_query_library)
            render_library_pair_summary(frame, rq4_pool_library)
            query_lang = frame[frame["library"] == rq4_query_library]["language"].iloc[0]
            pool_lang = frame[frame["library"] == rq4_pool_library]["language"].iloc[0]
            setting = "within-language" if query_lang == pool_lang else "cross-language"
            st.info(
                f"当前 RQ4 配置：测试集={rq4_query_library}，检索候选集={rq4_pool_library}，"
                f"类型={setting}。"
            )
    else:
        st.info("训练检索器时会统一使用全部 9 对数据集。当前数据对下拉框不影响训练范围。")

    with st.form("run_form"):
        data_dir = st.text_input("数据目录", value="data")
        output_root = st.text_input("输出根目录", value="outputs/ui_runs")
        device = st.selectbox("设备", ["cuda", "cpu"], index=0)

        c1, c2, c3 = st.columns(3)
        batch_size = c1.number_input("Batch Size", min_value=1, value=16, step=1)
        epochs = c2.number_input("Epochs", min_value=1, value=12, step=1)
        top_k = c3.number_input("Top-K", min_value=1, value=4, step=1)

        c4, c5, c6 = st.columns(3)
        grad_acc = c4.number_input("Grad Accumulation", min_value=1, value=1, step=1)
        seed = c5.number_input("Seed", min_value=0, value=42, step=1)
        max_queries = c6.number_input("Max Queries", min_value=0, value=0, step=1)

        model_name = st.text_input("模型名", value="deepseek-chat")
        trained_output_dir = ""
        api_key = ""
        if operation == "运行 RQ 实验":
            trained_output_dir = st.text_input("训练产物目录", value="outputs/train_run")
            api_key = st.text_input("DeepSeek API Key", value="", type="password")

        submitted = st.form_submit_button("开始执行", type="primary", use_container_width=True)

    command = "train" if operation == "训练检索器" else "rq"
    library_arg = None
    if operation == "运行 RQ 实验" and rq_id in {"RQ1", "RQ2", "RQ3"} and selected_library != "全部 9 对":
        library_arg = selected_library
    if operation == "运行 RQ 实验" and rq_id == "RQ4":
        run_token = f"{rq_id.lower()}_{rq4_query_library}_from_{rq4_pool_library}"
    else:
        run_token = f"{(rq_id or command).lower()}_{library_arg or 'all'}"
    run_name = f"{command}_{run_token}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = str(Path(output_root) / run_name)

    st.write("本次运行输出目录：", output_dir)
    if operation == "训练检索器":
        st.info("训练阶段统一用 9 对数据集训练检索器。")
    elif rq_id in {"RQ1", "RQ2", "RQ3"}:
        st.info(f"{rq_id} 现在是单独运行，不会顺带把其他 RQ 一起跑掉。")
    else:
        st.info("RQ4 现在是显式指定测试集 library 和检索候选集 library 的单独实验。")

    cmd = build_cli_command(
        command=command,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        batch_size=int(batch_size),
        epochs=int(epochs),
        top_k=int(top_k),
        grad_acc=int(grad_acc),
        seed=int(seed),
        model_name=model_name,
        library=library_arg,
        trained_output_dir=trained_output_dir or None,
        rq_id=rq_id,
        rq4_query_library=rq4_query_library,
        rq4_pool_library=rq4_pool_library,
        max_queries=int(max_queries),
        api_key=api_key,
    )

    with st.expander("调试命令", expanded=False):
        st.code(" ".join(cmd), language="bash")

    if submitted:
        with st.spinner("任务执行中..."):
            result = subprocess.run(
                cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
        st.session_state["last_output_dir"] = output_dir
        st.session_state["last_library"] = selected_library
        st.write("Return code:", result.returncode)
        if result.stdout:
            st.text_area("STDOUT", value=result.stdout, height=240)
        if result.stderr:
            st.text_area("STDERR", value=result.stderr, height=240)
        if result.returncode == 0:
            st.success("任务执行完成。结果已经写入上面的输出目录。")
        else:
            st.error("任务执行失败，请检查上面的输出。")


def render_results_tab(output_dir: Path) -> None:
    st.subheader("测试结果查看")
    active_output_dir = output_dir
    if "last_output_dir" in st.session_state:
        active_output_dir = Path(st.session_state["last_output_dir"])
        st.caption(f"最近一次运行目录：{active_output_dir}")
    else:
        st.caption(f"当前查看目录：{active_output_dir}")
    metrics_files = list_json_files(active_output_dir, "_metrics.json")
    prediction_files = list_json_files(active_output_dir, "_predictions.json")
    case_files = list_json_files(active_output_dir, "_cases.json")

    col1, col2, col3 = st.columns(3)
    col1.metric("metrics 文件数", len(metrics_files))
    col2.metric("predictions 文件数", len(prediction_files))
    col3.metric("cases 文件数", len(case_files))

    if metrics_files:
        selected_metrics = st.selectbox("选择 metrics 文件", metrics_files, format_func=lambda p: str(p.relative_to(active_output_dir)))
        metrics = load_json(selected_metrics)
        st.json(metrics, expanded=True)
    else:
        st.info("当前输出目录还没有 metrics 文件。")

    if prediction_files:
        selected_predictions = st.selectbox(
            "选择 predictions 文件",
            prediction_files,
            format_func=lambda p: str(p.relative_to(active_output_dir)),
        )
        predictions = pd.DataFrame(load_json(selected_predictions))
        st.dataframe(predictions, use_container_width=True, height=320)

    if case_files:
        selected_cases = st.selectbox("选择 case study 文件", case_files, format_func=lambda p: str(p.relative_to(active_output_dir)))
        cases = load_json(selected_cases)
        case_index = st.slider("案例索引", 0, max(len(cases) - 1, 0), 0)
        case = cases[case_index]
        st.markdown(f"**Query API**: `{case['query']['api']}`")
        st.markdown(f"**Query Label**: `{case['query']['label']}`")
        st.text_area("Query Fragment", value=case["query"]["fragment"], height=220)
        st.dataframe(pd.DataFrame(case["retrieved"]), use_container_width=True, height=280)


def main() -> None:
    st.set_page_config(page_title="HARK Test Console", layout="wide")
    st.title("HARK Test Console")
    st.caption("面向测试的数据总览、任务发起与结果查看界面")

    with st.sidebar:
        st.header("路径配置")
        data_dir = st.text_input("数据目录", value=str(ROOT / "data"))
        output_dir = st.text_input("输出目录", value=str(ROOT / "outputs"))

    frame = load_dataset_frame(data_dir)
    tab1, tab2, tab3 = st.tabs(["数据集", "发起测试", "结果查看"])
    with tab1:
        render_dataset_tab(frame)
    with tab2:
        render_runner_tab(frame)
    with tab3:
        render_results_tab(Path(output_dir))


if __name__ == "__main__":
    main()
