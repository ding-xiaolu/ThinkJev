# Third-party material

No model weights, third-party evaluation text, or thought traces are distributed here. Model
weights are downloaded from Hugging Face at the pinned revision; the bundled calibrations
(`src/thinkjev/calibrations/`) hold only temperatures, thresholds, and aggregate accuracies.
`benchmarks/manifests/` holds row IDs only, and the exporters in `benchmarks/` rebuild the text
from each upstream at a pinned revision; the license and revision of every evaluation source are
in [benchmarks/DATA_SOURCES.md](benchmarks/DATA_SOURCES.md). `benchmarks/data/` holds only
decisions generated for this project, by `benchmarks/generate_owned.py` (`owned-v1.jsonl`) and
`benchmarks/build_cascade.py hard` (`hard200.jsonl`). The optional `data` extra installs python-chess (GPL-3.0) to build the
chess decisions; the package never imports it.

| Item | Upstream | Pinned revision | Use |
|---|---|---|---|
| SemIf | https://github.com/TheoLeeCJ/SemIf | `1f2dea3e25379f9dfc98cb83c324f00ab5deda37` | Vendored, not installed: the S1 prompt, option readout, model loader, and prefix-cache helper in `src/thinkjev/_semif.py` (unchanged except that the cache helper forwards extra keyword arguments), and the temperature-scaling functions in `src/thinkjev/calibration.py`. MIT. |
| AnyJev | https://github.com/nokia-applied-research/AnyJev | `10d5db91dda38dbde74c6abc1c075ce6463723d1` | Cyclic-shift debiasing and coverage at risk informed the option rotations and the error-guarantee threshold; no code is copied. Apache-2.0. |
| Qwen3.5-4B | https://huggingface.co/Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Bundled calibration `qwen3.5-4b`; check the upstream model license. |
| Qwen3.5-9B | https://huggingface.co/Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Bundled calibration `qwen3.5-9b`; check the upstream model license. |
| Qwen3.5-27B | https://huggingface.co/Qwen/Qwen3.5-27B | `fc05daec18b0a78c049392ed2e771dde82bdf654` | Bundled calibration `qwen3.5-27b`; check the upstream model license. |
| Qwen3.8-27B | https://huggingface.co/Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Bundled calibration `qwen3.8-27b`. Apache-2.0. |
| TypeSafe System One API format | https://docs.typesafe.ai/api | Public documentation as of 2026-10-03 | `src/thinkjev/api.py` and `serve.py` accept and return the documented request and response shapes of `POST /v1/systemone` and compute confidence with the documented formulas, so TypeSafe's SDK can talk to a local ThinkJev server. No TypeSafe code, model, or output is included. |

ThinkJev is independent. It is not affiliated with or endorsed by TypeSafe, Jev, Every, or
the SemIf authors. Jev is TypeSafe AI's product; the name only says that ThinkJev answers
the same kind of typed decision. Named marks belong to their owners.

## SemIf license

MIT License

Copyright (c) 2026 TheoLeeCJ

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
