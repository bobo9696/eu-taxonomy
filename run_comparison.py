"""Run Baseline vs Hybrid evaluation comparison.

Usage:
    source apps/api/.venv/bin/activate
    python run_comparison.py

Output files (in current directory):
    - 01_comparison_aggregated.csv     – aggregated metrics per pipeline
    - 02_per_question_all_methods.csv    – per-question, all metrics, side-by-side

Column layout (per-question):
    Q# | Question | Ans? | Baseline Hit | Baseline Prec | ... | Hybrid Hit | Hybrid Prec | ...
"""

import asyncio
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "apps" / "api"))

from dotenv import load_dotenv
load_dotenv(Path.cwd() / "apps" / "api" / ".env")

# Warm up vector store + preload cross-encoder model in main thread
print("Preloading models...")
from rag.index.store import get_vector_store
get_vector_store()

from rag.query.methods.hybrid.cross_encoder_rerank import _ensure_loaded
_ensure_loaded()
print("Models ready.\n")

from rag.evaluate import evaluate_rag

configs = [
    {"name": "Baseline", "rag_type": "classic", "prompt_type": "intermediate", "use_rerank": False},
    {"name": "Baseline+Rerank", "rag_type": "classic", "prompt_type": "intermediate", "use_rerank": True},
    {"name": "Hybrid", "rag_type": "hybrid", "prompt_type": "intermediate", "use_rerank": False},
]

raw_results = []
agg_rows = []
for cfg in configs:
    print(f"⏳ Running {cfg['name']}...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res = loop.run_until_complete(
            evaluate_rag(rag_type=cfg["rag_type"], prompt_type=cfg["prompt_type"], use_rerank=cfg["use_rerank"])
        )
        loop.close()

        row = {"Pipeline": cfg["name"]}
        row.update(res["metrics"])
        agg_rows.append(row)
        raw_results.append({"name": cfg["name"], "samples": res["samples"]})
        print(f"   → retrieval_hit={row.get('retrieval_hit')}, faithfulness={row.get('faithfulness')}")
    except Exception as e:
        print(f"   ERROR: {e}")
        agg_rows.append({"Pipeline": cfg["name"], "ERROR": str(e)})
    print()

# ── 1. AGGREGATED COMPARISON ────────────────────────────────────────────
print("═" * 80)
print("  1) AGGREGATED METRICS per PIPELINE")
print("═" * 80)
agg_df = pd.DataFrame(agg_rows)
agg_df = agg_df.set_index("Pipeline")
agg_df.to_csv("01_comparison_aggregated.csv")
print(agg_df.T.to_string())
print("\n   📄 Saved → 01_comparison_aggregated.csv\n")

# ── 2. PER-QUESTION, ALL METHODS ────────────────────────────────────────
if len(raw_results) == 3:
    all_samples = [r["samples"] for r in raw_results]
    names = [r["name"] for r in raw_results]

    per_q_rows = []
    for i in range(len(all_samples[0])):
        row = {
            "Q#": i + 1,
            "Question": all_samples[0][i]["question"][:70] + ("…" if len(all_samples[0][i]["question"]) > 70 else ""),
            "Ans?": "Y" if all_samples[0][i]["answerable"] else "N",
        }
        for name, samples in zip(names, all_samples):
            scores = samples[i]["scores"]
            short = {
                "Baseline": "BL",
                "Baseline+Rerank": "BL+RR",
                "Hybrid": "HY",
            }[name]
            row[f"{short}_Hit"]   = round(scores.get("retrieval_hit", 0), 2)         if scores.get("retrieval_hit")   is not None else "-"
            row[f"{short}_Prec"]  = round(scores.get("context_precision", 0), 2)   if scores.get("context_precision") is not None else "-"
            row[f"{short}_Rec"]   = round(scores.get("context_recall", 0), 2)      if scores.get("context_recall")   is not None else "-"
            row[f"{short}_Faith"] = round(scores.get("faithfulness", 0), 2)         if scores.get("faithfulness")    is not None else "-"
            row[f"{short}_Rel"]   = round(scores.get("answer_relevancy", 0), 2)    if scores.get("answer_relevancy") is not None else "-"
        per_q_rows.append(row)

    per_q_df = pd.DataFrame(per_q_rows)
    per_q_df.to_csv("02_per_question_all_methods.csv", index=False)

    print("═" * 80)
    print("  2) PER-QUESTION, ALL METHODS & RAGAS METRICS")
    print("═" * 80)
    print("  Legend: BL=Baseline | BL+RR=Baseline+Rerank | HY=Hybrid")
    print("  Metrics: Hit=retrieval_hit | Prec=context_precision | Rec=context_recall | Faith=faithfulness | Rel=answer_relevancy")
    print()
    print(per_q_df.to_string(index=False))
    print("\n   📄 Saved → 02_per_question_all_methods.csv\n")
