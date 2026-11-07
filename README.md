# A Language-Model Prior for Cold-Start Prediction of AI Evaluation Outcomes

Code and forecasts behind our technical report for the Predictive AI Evaluation Competition (NeurIPS 2026). A language model reads an AI system's description and a benchmark item and states a success probability. That forecast is the offset of a hierarchical logistic model, which the revealed outcomes on the benchmark then update. The report's LaTeX source is in `report/paper/`.

## Contents

| Folder | What it holds |
|---|---|
| `submission/` | The predictor submitted to Codabench (submission 1027388). |
| `practice/` | The method, the forecasters' API calls, the exam builders and the evaluation code. |
| `report/figures/`, `analysis/` | The scripts behind the report's tables, figures and numbers. None of them calls a model. |
| `runs/` | Every forecast we collected, one folder per run, as JSON lines with the numbers and run metadata. |
| `scripts/` | Data download and exam building. |
| `report/paper/` | The report's LaTeX source. `tools/shade_tables.py` adds the table shading. |

Three things are left out on purpose.

- **The data and the exams built from it.** measurement-db is gated on Hugging Face, and the exams contain benchmark items. A public copy of the items would let them leak into future training data. The scripts below rebuild every exam, and each rebuild matched our original file exactly.
- **Text written by the models** (thinking summaries, agent transcripts, free-text rationales), because it can quote benchmark items. The released forecasts keep the probability and the run metadata, and `runs/thinking-opus-high-onepair/tags.jsonl` keeps the labels we derived from the thinking summaries.
- **API keys.**

## Setup

The scripts need [uv](https://docs.astral.sh/uv/). Run every command from the repository root, where the scripts find `practice/`, `runs/`, `exams/` and `data/` by relative path (`data/` and `exams/` are created by the steps below). Each script states its usage in its docstring. We ran everything on a laptop CPU with these versions, and every analysis script finishes within about 20 minutes:

```sh
PY="uv run --no-project --python 3.13 --with numpy==2.5.4 --with scipy==1.18.1 --with pandas==3.1.1 --with pyarrow==26.0.1 --with matplotlib==3.11.2 python -I"
```

## Get the data

1. **measurement-db at revision `54074453da7b9911be0f48e4c560be547bc84d21`** (about 3 GB of tables). Accept the dataset's terms on Hugging Face and log in with `huggingface-cli login`, then run

   ```sh
   huggingface-cli download aims-foundations/measurement-db --repo-type dataset \
     --revision 54074453da7b9911be0f48e4c560be547bc84d21 --include "*/formatted_tables/*" --local-dir data/hf
   echo 54074453da7b9911be0f48e4c560be547bc84d21 > data/hf/REVISION.txt
   ```

   `scripts/fetch_tables.py` and `scripts/fetch_small_tables.py` are the scripts we used to download parts of it at the same revision. The full download above covers everything the exams need.

2. **MathArena outputs**, for the check on problems written after the forecaster's training cutoff (CC BY-SA 4.0, pinned revisions, a few MB):

   ```sh
   uv run --no-project --with duckdb==2.0.0 python -I scripts/fetch_matharena.py data/matharena-1101
   ```

## Build the exams

```sh
sh scripts/build_exams.sh
```

This builds, under `exams/`, the five proxy evaluations (`rehearsal-02` and `proxy-confirm-1026` are the report's two validation proxies, `proxy-1028-1` to `-3` its three held-out proxies), their full acquisition pools, the benchmark catalog (`agent_box_02/`), the calibration set (`mdb-judge-1027`, 2,324 items from 332 subject-benchmark pairs on 121 benchmarks), its subset of one pair per benchmark (`mdb-judge-1027-onepair`, 847 items) and the MathArena exam. Its last step checks every exam file against the SHA-256 hashes in `scripts/exams.sha256`, which match our files byte for byte.

## Reproduce the report

The forecasts in `runs/` are the ones the report uses, so these scripts reproduce its numbers without any model call. Every row needs `data/` and `exams/` from the steps above. Each cell is a script with its arguments. Run it as `$PY <cell>`, with any variable in front, for example

```sh
FC_EXTRA=1 $PY report/figures/forecaster_constants.py 20
```

Scripts print their numbers to standard output, except the figure scripts, which write PDF files. Some printed labels predate the report's names: the "non-LLM rule" is the Beta-binomial baseline and the "hierarchical logit model with Opus offsets" is our method.

| Report | Command |
|---|---|
| Table 1 (all rows but the IRT model), Table 3 (ALC per proxy), per-proxy numbers in Appendix E | `report/figures/report_tables.py` |
| Table 1, IRT model row | `practice/irt_baseline.py fit <params.json> <exam_dirs>`, then `practice/baseline_eval.py` (arguments in their docstrings, fitted parameters in `runs/irt-baseline/params.json`) |
| Figure 1 | `pdflatex report/figures/fig_data.tex` |
| Figure 2 (worked example) and its paragraph in Section 3.3 | `report/figures/worked_example.py report/paper/figures` |
| Figures 3 and 4 | `report/figures/make_overview.py <out_dir>` |
| Table 2, Section 4.3 | `FC_EXTRA=1 report/figures/forecaster_constants.py 20` (constants and held-out ALC, with 20 bootstrap draws for the intervals of a and b), `analysis/recal_alone.py` (recalibrated forecast alone). Table 2 uses this default. Adding `FC_EXCLUDE_HELDOUT=1` refits the constants without the held-out benchmarks, the check in the report's Limitations. |
| Section 4.4, MathArena in Appendix E | `analysis/ma_update.py`, `analysis/ma_slope_sep.py`, `analysis/ma_slope_aug.py` (the August sets used to design the slope) |
| Table 4, Section 5.1 | `FC_EXTRA=1 analysis/gap_levels_family.py`, `analysis/matharena_eval.py`, `analysis/followups_1102.py` |
| Section 5.2, Table 8 | `practice/memory_generalization.py <out.csv>`, `report/figures/recognition_checks.py` |
| Section 5.3 | `practice/error_sharing.py exams/mdb-judge-1027 runs/mdb-judge-opus-low` |
| Section 5.4, Table 6 | `report/figures/hier_ablations.py`, `practice/acquisition_eval.py` |
| Table 5 (BLE) | `practice/ble_b0_compare.py`, `practice/ble_budget_compare.py` |
| Table 7 (effort) | `analysis/effort_pair.py`, `analysis/effort_claude_onepair.py` |

## Collect forecasts again

`practice/subscription_judge.py` asks Claude models through the Claude command-line interface, and `practice/openrouter_judge.py` asks other models through OpenRouter with a pinned provider, reading the OpenRouter key from a local file outside the repository (`KEY_FILE` in the script) without printing it. Model outputs vary between calls, so new forecasts will differ slightly from the ones in `runs/`.

## Run details not in the report

- **Forecast sources.** Claude forecasts came mostly from the Claude command-line interface and otherwise from the API (the first validation proxy's targets, the thinking summaries and BLE). When the interface returned nothing from the requested model it answered with an earlier model of the same line (Opus 4.8 or Opus 5, Sonnet 5): 1.7% of Opus's and 1.2% of Sonnet's calibration-set forecasts, and 0.8% of Opus's proxy forecasts. One refusal among the first validation proxy's remaining acquisition candidates was set to 0.575.
- **OpenRouter providers.** DeepInfra for DeepSeek V4.1 Flash, GMI Cloud for GLM-5.3 Flash and Baidu for GLM-5.3 (all fp8), and OpenAI for the GPT models (flex tier for GPT-6.1 Sol and Terra). GLM-5.3 Flash gave no answer within 24,000 tokens for 2% of held-out items, which fall back to 0.575.
- **Outcome-only IRT model.** Fitted values: mu = 0.28, standard deviations 1.67 for benchmarks, 0.73 for subjects within a benchmark and 1.22 for items (`runs/irt-baseline/params.json`).
- **Submitted predictor, live checks.** A live run on 24 predictions of the first validation proxy (budgets 15 and 31) took at most 7 seconds per call. On its biology benchmark Opus refused 20 of 40 requests, and the second request recovered the target in half of the calls whose first request was refused. The formative run's API calls cost $56. A single API forecast from Opus costs about $0.010.

## Run the submitted predictor

Copy `submission/submission_config.example.json` to `submission/submission_config.json`, put an Anthropic API key in it, and package `submission/` as the competition's starting kit describes. That file is listed in `.gitignore` so the key is not committed.

## License

The code is released under the MIT License. measurement-db and MathArena keep their own licenses (CC BY-SA 4.0).
