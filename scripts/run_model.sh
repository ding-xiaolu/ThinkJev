#!/usr/bin/env bash
# Score one model on the release rows: S1 (+ rotations, features, pre-fork features) and the router
# readout on one set of GPUs, vLLM thoughts on another, then the S2 readout, then merge.
#
#   scripts/run_model.sh TAG MODEL REVISION BUDGET "S1_GPUS" "THINK_GPUS" [ROWS_DIR]
#   e.g. scripts/run_model.sh qwen3.5-4b Qwen/Qwen3.5-4B <40-char sha> 4096 "0 1" "0 1 2 3"
#
# S1_GPUS and THINK_GPUS are CUDA device indices on this machine (one process per GPU; the two lists
# may be the same GPUs only if they run on different machines — run the halves separately with
# STAGE=s1 / STAGE=think / STAGE=readout / STAGE=merge when splitting the work across machines).
# Environments: .venv (this package) and .venv-vllm (vLLM). Outputs are create-only: the script
# refuses to reuse a run directory. Final outputs go to results/ only when every shard finished.
set -euo pipefail

TAG=$1 MODEL=$2 REVISION=$3 BUDGET=$4
read -r -a S1_GPUS <<< "$5"
read -r -a THINK_GPUS <<< "$6"
ROWS_DIR=${7:-build/release}
STAGE=${STAGE:-all}
RUN=build/runs/$TAG-b$BUDGET
ROWS=$ROWS_DIR/rows.jsonl
ROUTER=$ROWS_DIR/router.jsonl
LAYERS=${FEATURE_LAYERS:-16,20,24,28,32}
CHECKPOINTS=${CHECKPOINTS:-0,128,256,512,1024,2048}  # anytime readouts: answers under these think budgets
MAIN=${MAIN_VENV:-.venv}/bin
VLLM=${VLLM_VENV:-.venv-vllm}/bin

split_rows() {  # split FILE N PREFIX: order-preserving line split
  split -n "l/$2" -d -a 2 --additional-suffix=.jsonl "$1" "$3"
}

stage() { [[ $STAGE == all || $STAGE == "$1" ]]; }

if stage s1; then
  mkdir -p "$(dirname "$RUN")" && mkdir "$RUN"  # fails if the run exists: outputs are create-only
  n=${#S1_GPUS[@]}
  split_rows "$ROWS" "$n" "$RUN/rows-"
  [[ ! -e $ROUTER ]] || split_rows "$ROUTER" "$n" "$RUN/router-"
  "$MAIN/thinkjev-s2" prompts --model "$MODEL" --revision "$REVISION" --input "$ROWS" \
    --output "$RUN/prompts.jsonl"
  for i in "${!S1_GPUS[@]}"; do
    s=$(printf %02d "$i")
    (
      CUDA_VISIBLE_DEVICES=${S1_GPUS[$i]} "$MAIN/thinkjev-score" --s2 never --model "$MODEL" \
        --revision "$REVISION" --input "$RUN/rows-$s.jsonl" --output "$RUN/s1-$s.jsonl" --rotations \
        --features "$RUN/features-$s.npy" --prefork-features "$RUN/prefork-$s.npy" --feature-layers "$LAYERS"
      if [[ -e $RUN/router-$s.jsonl ]]; then
        CUDA_VISIBLE_DEVICES=${S1_GPUS[$i]} "$MAIN/thinkjev-score" --mode direct --model "$MODEL" \
          --revision "$REVISION" --input "$RUN/router-$s.jsonl" --output "$RUN/routerout-$s.jsonl"
      fi
    ) > "$RUN/s1-$s.log" 2>&1 &
  done
  wait
fi

if stage think; then
  n=${#THINK_GPUS[@]}
  split_rows "$RUN/prompts.jsonl" "$n" "$RUN/prompts-"
  for i in "${!THINK_GPUS[@]}"; do
    s=$(printf %02d "$i")
    CUDA_VISIBLE_DEVICES=${THINK_GPUS[$i]} "$VLLM/python" benchmarks/vllm_think.py --model "$MODEL" \
      --revision "$REVISION" --prompts "$RUN/prompts-$s.jsonl" --output "$RUN/thoughts-$s.jsonl" \
      --think-budget "$BUDGET" --seeds "${SEEDS:-1}" > "$RUN/think-$s.log" 2>&1 &
  done
  wait
  cat "$RUN"/thoughts-[0-9][0-9].jsonl > "$RUN/thoughts.jsonl"
fi

if stage readout; then
  for i in "${!S1_GPUS[@]}"; do
    s=$(printf %02d "$i")
    CUDA_VISIBLE_DEVICES=${S1_GPUS[$i]} "$MAIN/thinkjev-s2" readout --model "$MODEL" --revision "$REVISION" \
      --input "$RUN/rows-$s.jsonl" --s1 "$RUN/s1-$s.jsonl" --thoughts "$RUN/thoughts.jsonl" \
      --prompts "$RUN/prompts.jsonl" --output "$RUN/cascade-$s.jsonl" --checkpoints "$CHECKPOINTS" \
      > "$RUN/readout-$s.log" 2>&1 &
  done
  wait
fi

if stage merge; then
  total=$(wc -l < "$ROWS")
  parts=(cascade); [[ ! -e $ROUTER ]] || parts+=(routerout)
  for part in "${parts[@]}"; do
    cat "$RUN"/$part-[0-9][0-9].jsonl > "$RUN/$part.jsonl"
    got=$(wc -l < "$RUN/$part.jsonl")
    [[ $got == "$total" ]] || { echo "$part has $got of $total rows; rerun the failed shard" >&2; exit 1; }
  done
  cat "$RUN"/features-[0-9][0-9].npy > "$RUN/features.npy"
  cat "$RUN"/prefork-[0-9][0-9].npy > "$RUN/prefork.npy"
  for target in "results/$TAG-b$BUDGET-cascade.jsonl" "results/$TAG-router.jsonl"; do
    [[ ! -e $target ]] || { echo "$target exists; results are create-only" >&2; exit 1; }
  done
  mkdir -p results
  cp "$RUN/cascade.jsonl" "results/$TAG-b$BUDGET-cascade.jsonl"
  [[ ! -e $RUN/routerout.jsonl ]] || cp "$RUN/routerout.jsonl" "results/$TAG-router.jsonl"
  echo "merged $total rows into results/$TAG-b$BUDGET-cascade.jsonl"
fi
