# service.py
from typing import List, Dict, Any, Tuple
import mlflow

PRICE = {
    "input_per_million": 0.10,  # adjust if using pro model, see price on official website
    "output_per_million": 0.40,
}

def _cost(in_tokens: int, out_tokens: int, price: Dict[str, float] = PRICE) -> Tuple[float, float, float]:
    ic = (in_tokens / 1_000_000) * price["input_per_million"]
    oc = (out_tokens / 1_000_000) * price["output_per_million"]
    return ic, oc, ic + oc

def build_messages(query: str, facts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    blocks = []
    for x in facts:
        topic = x.get("Topic", "N/A")
        content = x.get("Content", "")
        blocks.append(f"[{topic}]\n{content}")
    doc = "\n\n---\n\n".join(blocks)
    prompt = (
        "Summarize for a technical reader.\n\n"
        f"Query:\n{query}\n\n"
        "Facts:\n"
        f"{doc}\n\n"
        "One short paragraph. No lists. No extra knowledge."
    )
    return [
        {"role": "system", "content": "summarize technical content"},
        {"role": "user", "content": prompt},
    ]

class Summarizer:
    def __init__(self, adapter, experiment: str, price: Dict[str, float] = PRICE):
        self.adapter = adapter
        self.experiment = experiment
        self.price = price
        mlflow.set_experiment(experiment)

    def run(self, query: str, facts: List[Dict[str, Any]], temperature: float = 0.4) -> Dict[str, Any]:
        msgs = build_messages(query, facts)
        res = self.adapter.complete(msgs, temperature=temperature)
        in_t, out_t = int(res.get("in_tokens", 0)), int(res.get("out_tokens", 0))
        ic, oc, tc = _cost(in_t, out_t, self.price)

        with mlflow.start_run() as run:
            mlflow.log_param("model", getattr(self.adapter, "model", "unknown"))
            mlflow.log_param("temperature", temperature)
            mlflow.log_param("query", query)

            mlflow.log_metric("input_tokens", in_t)
            mlflow.log_metric("output_tokens", out_t)
            mlflow.log_metric("latency_s", float(res.get("latency", 0.0)))
            mlflow.log_metric("input_cost_usd", ic)
            mlflow.log_metric("output_cost_usd", oc)
            mlflow.log_metric("total_cost_usd", tc)

            mlflow.log_text(msgs[0]["content"], "system_prompt.txt")
            mlflow.log_text(msgs[-1]["content"], "user_prompt.txt")
            mlflow.log_text(res.get("text", ""), "assistant_response.txt")

            run_id = run.info.run_id

        return {
            "text": res.get("text", ""),
            "run_id": run_id,
            "cost_total_usd": tc,
        }
