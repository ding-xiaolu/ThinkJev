"""Print the current commit of every dataset and model we pin, so it can be recorded.

Run where Hugging Face and ModelScope are reachable, in the data environment:

    build/venv-data/bin/python benchmarks/resolve_revisions.py > build/revisions.tsv

Hugging Face SHAs come from `huggingface_hub.HfApi().dataset_info(name).sha` and
`model_info(name).sha`. ModelScope dataset revisions (EvalScope's default hub) are looked up
best effort; when no commit ID is exposed, pin by the SHA-256 that `export_evalscope.py`
writes next to each export. Names that fail print an ERROR line instead of stopping.
Copy the SHAs into THIRD_PARTY.md (models) or benchmarks/DATA_SOURCES.md (data) and pass them as `--revision`.
"""

from __future__ import annotations

import sys

# export_hf.py sources (Hugging Face).
HF_DATASETS = ["mteb/banking77", "clinc/clinc_oos", "LocalLLaMA/typed-decisions", "allenai/reward-bench-2"]
# EvalScope 1.12.0 dataset IDs for export_evalscope.py (checked against the registry when installed).
EVALSCOPE_DATASETS = {
    "bbh": "evalscope/bbh",
    "mmlu_pro": "TIGER-Lab/MMLU-Pro",
    "gpqa_diamond": "AI-ModelScope/gpqa_diamond",
    "super_gpqa": "m-a-p/SuperGPQA",
    "logi_qa": "extraordinarylab/logiqa",
    "musr": "AI-ModelScope/MuSR",
    "ceval": "evalscope/ceval",
    "cmmlu": "evalscope/cmmlu",
    "agieval": "opencompass/agieval",
}
# Candidate models; fix a name here if it is not found.
MODELS = [
    "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "HuggingFaceTB/SmolLM3-3B",
]


def emit(kind: str, hub: str, name: str, value: str) -> None:
    print(f"{kind}\t{hub}\t{name}\t{value}", flush=True)


def modelscope_revision(api, name: str) -> str:
    info = api.get_repo(name, "dataset")
    for attribute in ("sha", "commit_id", "last_commit_id", "revision"):
        value = getattr(info, attribute, None)
        if value:
            return str(value)
    last = getattr(info, "last_commit", None) or getattr(info, "LastCommit", None)
    if last:
        return str(getattr(last, "id", None) or getattr(last, "Id", None) or last)
    return f"no commit id exposed ({type(info).__name__}); pin by export SHA-256"


def main() -> None:
    try:
        from evalscope.api.registry import BENCHMARK_REGISTRY
        import evalscope.benchmarks  # noqa: F401  (registers adapters)

        for benchmark, dataset in EVALSCOPE_DATASETS.items():
            actual = BENCHMARK_REGISTRY[benchmark].dataset_id
            if actual != dataset:
                emit("WARNING", "evalscope", benchmark, f"registry dataset_id is {actual}, table says {dataset}")
    except ImportError:
        print("# evalscope not installed; EvalScope dataset IDs are unchecked", file=sys.stderr)

    from huggingface_hub import HfApi

    api = HfApi()
    for name in HF_DATASETS + [dataset for dataset in EVALSCOPE_DATASETS.values()]:
        try:
            emit("dataset", "huggingface", name, api.dataset_info(name).sha)
        except Exception as error:  # noqa: BLE001  keep going; report every failure
            emit("ERROR", "huggingface", name, f"{type(error).__name__}: {error}".replace("\n", " ")[:200])
    for name in MODELS:
        try:
            emit("model", "huggingface", name, api.model_info(name).sha)
        except Exception as error:  # noqa: BLE001
            emit("ERROR", "huggingface", name, f"{type(error).__name__}: {error}".replace("\n", " ")[:200])

    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        print("# modelscope not installed; ModelScope revisions skipped", file=sys.stderr)
        return
    ms_api = HubApi()
    for name in EVALSCOPE_DATASETS.values():
        try:
            emit("dataset", "modelscope", name, modelscope_revision(ms_api, name))
        except Exception as error:  # noqa: BLE001
            emit("ERROR", "modelscope", name, f"{type(error).__name__}: {error}".replace("\n", " ")[:200])


if __name__ == "__main__":
    main()
