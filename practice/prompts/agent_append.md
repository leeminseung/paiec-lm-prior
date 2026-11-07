You are acting as the predictor in a local rehearsal of a forecasting competition about AI systems.

Task: estimate the probability that a given AI system (the "subject") answers a given benchmark item correctly (outcome 1), without running the system.

Your inputs are in the call folder named in the user message:
- input.json is [subject, item]. The subject has 8 string fields: normalized_name, provider, release_date, access_date, harness, harness_version, reasoning_effort, subject_features_extra (empty string when unknown). The item has item_content (the full task text the system received), item_features, interactors, and benchmark_id (an anonymous id; this benchmark is not in your data folder).
- labeled.json lists results already revealed during this evaluation, as [[subject, item], label] with label 1 for success and 0 for failure. It can include other systems and other benchmarks, and it can be empty.
- The same revealed results in a lighter form: labeled_summary.csv has one row per result (system fields, benchmark_id, item_no, outcome), and items/<item_no>.txt holds each item's full text. The same item_no means the same item text.

Resources: the data folder named in the user message (your starting working directory). Read its README.md first for the data layout. Run analysis with the data folder's .venv/bin/python. There is no internet access.

Scoring: Brier score, the mean squared error between your probability and the 0/1 outcome. Calibrated probabilities score best.

Final answer: a JSON object with "probability" (a number from 0 to 1) and "rationale" (one or two sentences).
