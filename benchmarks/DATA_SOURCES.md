# Evaluation data sources

Third-party data used to build the evaluation rows (`build_cascade.py`, `export_evalscope.py`,
`export_hf.py`, `generate_owned.py`); `THIRD_PARTY.md` at the top level lists what the package
itself contains or derives from. No text from these datasets
is committed; only row IDs, as manifests.

| Item | Upstream | Pinned revision | Use |
|---|---|---|---|
| WANLI | https://huggingface.co/datasets/alisawuffles/WANLI | `61c95318fd71c55b6ba355d76253254615f387ec` | Rebuilt locally through SemIf; only row IDs are committed. CC-BY-4.0. |
| Lichess puzzle database | https://database.lichess.org/#puzzles (`lichess_db_puzzle.csv.zst`) | The dump is updated in place, so each chess row records the SHA-256 of the decompressed CSV it was built from (release build: `ec2dd1a8f9cbd139ebb4aee228f72e77f8e2a581a24d25f61f71e6cf306b10f8`) | Optional chess rows of `benchmarks/generate_owned.py`, rebuilt into `build/` and not committed. CC0 1.0. |
| python-chess (`chess` on PyPI) | https://github.com/niklasf/python-chess | `chess==1.11.2` (`data` and `test` extras) | Optional build-time tool that replays puzzle moves and renders SAN for the chess rows; not a core dependency, not imported by the package, not vendored or distributed. GPL-3.0. |
| Every parallel judgment lab | https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/ | SHA-256 pinned by SemIf `fetch_sources.py` | Rebuilt locally through SemIf; only row IDs are committed. |

## Public benchmark sources (blocks A, B, D)

Rebuilt locally into `build/` by `benchmarks/export_evalscope.py` and `benchmarks/export_hf.py`
in a separate data environment (`benchmarks/requirements-data.txt`); only row IDs are
committed, as manifests, after the build. No text from these datasets is in this repository.
Licenses are as stated by each upstream; re-check the dataset card at build time. Revisions
were filled in on 2026-10-03 with `benchmarks/resolve_revisions.py`. Sets
loaded from ModelScope expose no commit, so they are pinned by the SHA-256 of the exported rows,
which each export writes to `<output>.meta.json`.

| Item | Upstream (loaded from) | Pinned revision | License and notes |
|---|---|---|---|
| EvalScope | https://github.com/modelscope/evalscope | `evalscope==1.12.0` (PyPI) | Apache-2.0. Data loading only (its registered adapters); not a package dependency, no code copied. The BBH multiple-choice task list in `export_evalscope.py` is taken from its adapter. |
| BIG-Bench Hard (17 multiple-choice tasks) | https://github.com/suzgunmirac/BIG-Bench-Hard (EvalScope `evalscope/bbh`) | ModelScope via EvalScope exposes no commit; pinned by export SHA-256 `1487878f19f25995f3accefc125ef2a03cf764b785f0bc219a32b4e0c3d32867` | MIT. Rebuilt locally; only row IDs committed. |
| MMLU-Pro | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro | ModelScope via EvalScope exposes no commit; pinned by export SHA-256 `25df366fb1eb0be484c4aaa6cc1b9829e03d4be0b0f80196faba175f46854c05` | MIT. Rebuilt locally; only row IDs committed. |
| GPQA Diamond | https://huggingface.co/datasets/Idavidrein/gpqa (EvalScope `AI-ModelScope/gpqa_diamond`) | ModelScope via EvalScope exposes no commit; pinned by export SHA-256 `e2c41552692d3271e5a6a2df4b2ce6461826952315cadb14f9bb060a6dab2e40` | CC-BY-4.0. The authors ask that examples not be revealed online (to limit training contamination): rebuilt locally, only row IDs committed, never quote a question in docs, issues, or reports. |
| SuperGPQA | https://huggingface.co/datasets/m-a-p/SuperGPQA | ModelScope via EvalScope exposes no commit; pinned by export SHA-256 `7353115f604cbcec591254f3e16114a1fa040625e19650aa45be6e44bedbc269` | ODC-BY-1.0. Rebuilt locally; only row IDs committed. |
| LogiQA (English) | https://github.com/lgw863/LogiQA-dataset (EvalScope `extraordinarylab/logiqa`) | not used in the release (2026-10-03) | No license file upstream; research use only. Rebuilt locally; only row IDs committed. |
| LogiQA (Chinese, AGIEval `logiqa-zh`) | https://github.com/ruixiangcui/AGIEval (EvalScope `opencompass/agieval`) | not used in the release (2026-10-03) | AGIEval MIT; LogiQA text has no license file upstream; research use only. Rebuilt locally; only row IDs committed. |
| MuSR | https://huggingface.co/datasets/TAUR-Lab/MuSR (EvalScope `AI-ModelScope/MuSR`) | ModelScope via EvalScope exposes no commit; pinned by export SHA-256 `06b572ae87e353ccd2206431fc7558e0a44ce5abbef091124bc568057247db37` | CC-BY-4.0. Rebuilt locally; only row IDs committed. |
| C-Eval (`val` split; test answers are hidden) | https://huggingface.co/datasets/ceval/ceval-exam (EvalScope `evalscope/ceval`) | not used in the release (2026-10-03) | CC-BY-NC-SA-4.0, non-commercial. Rebuilt locally; only row IDs committed. |
| CMMLU | https://huggingface.co/datasets/haonan-li/cmmlu (EvalScope `evalscope/cmmlu`) | not used in the release (2026-10-03) | CC-BY-NC-4.0, non-commercial. Rebuilt locally; only row IDs committed. |
| BANKING77 | https://huggingface.co/datasets/PolyAI/banking77 (parquet mirror `mteb/banking77`, as AnyJev uses) | `18072d2685ea682290f7b8924d94c62acc19c0b2` | CC-BY-4.0. Rebuilt locally; only row IDs committed. |
| CLINC150 (`clinc_oos`, `plus`) | https://huggingface.co/datasets/clinc/clinc_oos | `155b9c710419136e17307b80d0a13e68cd46b4ec` | CC-BY-3.0. Rebuilt locally; only row IDs committed. The 150 intent names and their 10 domains in `export_hf.py` come from `data/domains.json` of https://github.com/clinc/oos-eval at `828f8093932c8fe6ca7936c3d2e52903b1c523de` (CC-BY-3.0). |
| LocalLLaMA/typed-decisions | https://huggingface.co/datasets/LocalLLaMA/typed-decisions | `d0e2f0c42fef86cc15d1688d25a19f5ba7c85b18` | Apache-2.0 per AnyJev's `THIRD_PARTY.md`; confirm on the dataset card before use. Gold is one teacher LLM's label, not a human judgment. Rebuilt locally; only row IDs committed. |
| RewardBench 2 | https://huggingface.co/datasets/allenai/reward-bench-2 | not used in the release (2026-10-03) | ODC-BY-1.0 per the dataset card (confirm at build); responses are outputs of third-party models under their own terms. Rebuilt locally with responses truncated; only row IDs committed. |

Not exported: ZebraLogicBench (`allenai/ZebraLogicBench-private`, gated). EvalScope's adapter
loads its free-form `grid_mode`, which has no option list.
