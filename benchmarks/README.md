# Reproducing the technical report

Everything the report measures can be checked at two depths: from the released outputs on a CPU
in minutes, or from scratch on GPUs. Run every command from the repository root.

## Environments

```bash
# Main: scorer, evaluation, figures, tests
python -m venv .venv && .venv/bin/pip install -e '.[test,figures]' && .venv/bin/pytest -q
# vLLM samples thoughts only; it pins its own torch, so it gets its own environment
python -m venv .venv-vllm && .venv-vllm/bin/pip install vllm==0.30.0
# Data: the public-source exporters (EvalScope, datasets; no torch)
python -m venv build/venv-data && build/venv-data/bin/pip install -r benchmarks/requirements-data.txt
```

Optional on CUDA: `pip install flash-linear-attention causal-conv1d` in `.venv` for fast Gated
DeltaNet kernels. They can change BF16 numerics slightly, so use them in every run or in none.

## From the released outputs (CPU)

The per-decision outputs and reports of the release runs are attached to the GitHub release as
`results.tar.gz`. They hold IDs, logits, token counts, and timings, never decision text.

```bash
tar -xzf results.tar.gz                      # writes results/
(cd results && shasum -a 256 -c SHA256SUMS)
for ml in qwen3.5-4b:Qwen3.5-4B qwen3.5-9b:Qwen3.5-9B qwen3.5-27b:Qwen3.5-27B qwen3.8-27b:Qwen3.8-27B; do
  m=${ml%%:*}
  .venv/bin/python benchmarks/make_figures.py --report results/$m-b4096-report-v2.json \
    --label ${ml#*:} --out build/figures/$m \
    --manifest benchmarks/manifests/release-split.jsonl \
    --jevbench results/jevbench-$m-report-v2.json
done
```

Add `--latency results/qwen3.5-4b-latency.jsonl` for Qwen3.5-4B. Each output directory has the
figures, `summary.md` (every number names the report field it comes from), and the LaTeX number
and table files of the report; apart from comment headers they equal `results/figures/<model>/`.

## From scratch (GPU)

**1. Decisions.** No third-party text is committed, so the rows are rebuilt from each upstream at
the revision in [DATA_SOURCES.md](DATA_SOURCES.md):

```bash
# SemIf's authored, WANLI, and Every rows
git clone https://github.com/TheoLeeCJ/SemIf.git build/semif
git -C build/semif checkout 1f2dea3e25379f9dfc98cb83c324f00ab5deda37
python build/semif/benchmarks/fetch_sources.py --output build/sources
python build/semif/benchmarks/build_wanli.py --source build/sources/wanli-test.jsonl \
  --selection build/semif/benchmarks/manifests/source-selection.jsonl --output build/gold-wanli256.jsonl
python build/semif/benchmarks/build_every.py --archive build/sources/every-source.zip \
  --experiments build/sources/every-experiments.json \
  --selection build/semif/benchmarks/manifests/source-selection.jsonl --output-dir build/every

# Public benchmarks (data environment); $SHA_* are the revisions in DATA_SOURCES.md
for s in bbh mmlu_pro gpqa_diamond super_gpqa musr; do
  build/venv-data/bin/python benchmarks/export_evalscope.py --source $s --output build/public/$s.jsonl
done
build/venv-data/bin/python benchmarks/export_hf.py --source banking20 --revision $SHA_BANKING77 --output build/public/banking20.jsonl
for s in clinc_escalate clinc_intent; do
  build/venv-data/bin/python benchmarks/export_hf.py --source $s --revision $SHA_CLINC --output build/public/$s.jsonl
done
build/venv-data/bin/python benchmarks/export_hf.py --source typed_decisions --revision $SHA_TYPED --output build/public/typed_decisions.jsonl

# Chess decisions from the Lichess puzzle database (CC0)
.venv/bin/pip install -e '.[data]'
curl -L -o build/lichess_db_puzzle.csv.zst https://database.lichess.org/lichess_db_puzzle.csv.zst
zstd -d build/lichess_db_puzzle.csv.zst -o build/lichess_db_puzzle.csv
.venv/bin/python benchmarks/generate_owned.py --families chess --chess-csv build/lichess_db_puzzle.csv \
  --output build/owned-chess.jsonl
```

The generated decisions are committed in `data/` (`owned-v1.jsonl`, `hard200.jsonl`). Assemble
everything and check the split against the committed manifest:

```bash
.venv/bin/python benchmarks/build_cascade.py assemble \
  --source authored=build/semif/benchmarks/data/authored144.jsonl --source wanli=build/gold-wanli256.jsonl \
  --source every=build/every/gold154.jsonl --source hard=benchmarks/data/hard200.jsonl \
  --source owned=benchmarks/data/owned-v1.jsonl --source chess=build/owned-chess.jsonl \
  $(for s in bbh mmlu_pro gpqa_diamond super_gpqa musr banking20 clinc_escalate clinc_intent typed_decisions; do
      echo --source $s=build/public/$s.jsonl; done) \
  --output-dir build/release --manifest build/release-split.jsonl
cmp build/release-split.jsonl benchmarks/manifests/release-split.jsonl   # must be identical
```

The Lichess dump is updated in place; a chess decision whose puzzle has changed upstream will not
match its manifest entry, and DATA_SOURCES.md records how those rows were pinned.

**2. Model runs.** One command per model, with the revisions in the report's appendix. It shards
rows across GPUs, one GPU per process, and writes to `results/` only when every shard finished:

```bash
scripts/run_model.sh qwen3.5-4b Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a 4096 "0 1" "2 3 4 5"
```

The first GPU list scores the direct answer and the router readout; the second samples thoughts
with vLLM. `scripts/run_model.sh` explains how to split the stages across machines.
`benchmarks/check_two_stage.py` checks, on a few rows, that the batch readout reproduces the
one-process scorer.

**3. Evaluation.** The hidden-state features are written to the run directory, not committed:

```bash
RUN=build/runs/qwen3.5-4b-b4096
.venv/bin/python benchmarks/evaluate_cascade.py --rows build/release/rows.jsonl \
  --cascade results/qwen3.5-4b-b4096-cascade.jsonl --router results/qwen3.5-4b-router.jsonl \
  --features $RUN/features.npy --prefork-features $RUN/prefork.npy \
  --report build/qwen3.5-4b-b4096-report.json
```

then `make_figures.py` as above. Latency: `benchmarks/sample_rows.py --per-stratum 40` on the
release rows, then `thinkjev-score --s2 always --think-budget 4096 --rotations` on one otherwise
idle GPU.

**JevBench check.** `export_jevbench.py` turns the public items of a JevBench checkout at commit
`7625ecb1b075cfe4d09d34141d353325825eea1d` into decisions (IDs in
`manifests/jevbench-public.jsonl`); they run through `scripts/run_model.sh` unchanged, and
`evaluate_jevbench.py` compares each model with JevBench's published per-task outcomes.
