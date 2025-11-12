# main.py
import os
import json
from adapter import GeminiAdapter
from service import Summarizer
from report import create_text_log_report  # <— add this line

EXPERIMENT = "/ryan/AI_Summarization_Pipeline"
FACTS_FILE = "facts.json"
QUERY = "Summarize the core technologies and risks of Large Language Models."

def _get_keys():
    try:
        from google.colab import userdata  # type: ignore
        api = userdata.get("GEMINI_API_KEY")
        model = userdata.get("MODEL_NAME")
        if api:
            os.environ["GEMINI_API_KEY"] = api
    except Exception:
        api = os.environ.get("GEMINI_API_KEY")
        model = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    return api, model

def run():
    api, model = _get_keys()
    if not api:
        print("no api key")
        return

    with open(FACTS_FILE, "r", encoding="utf-8") as f:
        facts = json.load(f)

    adapter = GeminiAdapter(api_key=api, model=model)
    svc = Summarizer(adapter, experiment=EXPERIMENT)
    out = svc.run(query=QUERY, facts=facts, temperature=0.4)

    print("model:", model)
    print("facts:", len(facts))
    print("\nsummary:\n")
    print(out["text"])
    print("\nrun id:", out["run_id"])
    print("cost (usd):", f"{out['cost_total_usd']:.6f}")

    # generate detailed text log
    create_text_log_report(EXPERIMENT, "llm_run_report.txt")
