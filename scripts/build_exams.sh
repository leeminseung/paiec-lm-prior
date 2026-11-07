#!/bin/sh
# Rebuild every exam used in the report from measurement-db (revision 54074453) and the MathArena outputs.
# Run from the repository root after downloading the data (README). Each rebuild matched the original exam exactly.
set -e
PY="uv run --no-project --python 3.13 --with pandas==3.1.1 --with pyarrow==26.0.1 --with numpy==2.5.4 python -I"
PYD="uv run --no-project --python 3.13 --with pandas==3.1.1 --with pyarrow==26.0.1 --with numpy==2.5.4 --with duckdb==2.0.0 python -I"
$PY practice/make_exam.py data/hf exams/rehearsal-02 20261020 "medhelm_medcalc_bench|f8cf1fbfad113ad7|110,workarena|ffe281c5769e382b|100,benger|b39fad85f451e932|89,openbiorq|c38c0f5d02555398|133,swe_rebench|58c3de46be1beb60|102,robust_reasoning_benchmark|7dbbe2b0319dff07|94,sgrades|6efd3a3caa2ebfba|121,openbiorq|7537722f6d86b119|106,benger|8ea91b5297ebb32e|121"
$PY practice/make_exam.py data/hf exams/proxy-confirm-1026 20261026 "appworld|0c300c3903b32b17|102,kernelbench|b676c2ae32416bc6|114,medhelm_medec|34d0cf520d687799|95,babilong|abc613dafba5b3dd|116,algotune|49eb2bd21bc77c78|127,dave|9e410cbb4427d073|93,worldcentralbanks|4198af63aedae22f|128,babilong|587f60939c893780|100,algotune|676ede36cce18bfa|125"
$PY practice/make_exam.py data/hf exams/proxy-1028-1 202610281 "real_webagents|7fcaa05b8c908c9f|94,edu_circuit_hw|c535a2a66f0007ef|118,tau2_bench|b4c14bc521124cd4|118,software_world_models|2f20ea9e0c2bf3b1|118,mathconstraint|62164bb2b6e485da|118,agentic_review_perturbation|a6934833d7795a4c|91,sketchjudge|0d44677761cab528|119,software_world_models|0e498e0a5b1ac8f8|119,tau2_bench|51f0eb6d336b192f|105"
$PY practice/make_exam.py data/hf exams/proxy-1028-2 202610282 "medhelm_medhallu|f8cf1fbfad113ad7|120,workbench_revisited|92904a3bade4a0ca|120,mmlupro|a7eae8f7c3f50f03|120,medhelm_medbullets|bb36678a69792bdd|99,agi_elo|ac44818ea0963843|120,risebench|d93fcb47e7bd7923|114,healthadminbench|7fd8baeee1896835|103,medhelm_medbullets|ec6c70f92d28548e|111,risebench|84bf060a65ae8177|93"
$PY practice/make_exam.py data/hf exams/proxy-1028-3 202610283 "multi_swebench|724aee89f8778c84|134,faithcot|bfbac93344b0c1a7|111,visualwebarena|2cb966883dff3c9c|134,legal_rag_bench|73cf38b394602dd8|89,hle|ebaa832466f958e7|80,icpc2_code_selector|d448483c7b3147bf|134,naturebench|fb20a2c364453a0d|90,legal_rag_bench|145f4adb93ddabcc|105,multi_swebench|b8dc37f628b47076|119"
$PY practice/extend_pool.py data/hf exams/rehearsal-02 exams/proxy-confirm-1026 exams/proxy-1028-1 exams/proxy-1028-2 exams/proxy-1028-3
$PYD practice/build_agent_box.py data/hf exams/rehearsal-02 agent_box_02   # benchmark catalog (hard links, no copy)
$PY practice/build_indexes.py agent_box_02
$PY practice/make_judge_set.py exams/mdb-judge-1027 20261027 3 7
$PY practice/subset_exam.py exams/mdb-judge-1027 exams/mdb-judge-1027-onepair 1
$PY practice/make_matharena_exam.py data/matharena-1101 exams/matharena-1101 prompt
# every exam file should match ours byte for byte
shasum -a 256 -c scripts/exams.sha256
