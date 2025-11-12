import mlflow

# simple price table kept here
PRICE = {
    "input_per_million": 0.10,   # adjust if you use pro model, check the price on the official website
    "output_per_million": 0.40,
}

def _cost(in_tokens, out_tokens):
    ic = (in_tokens / 1_000_000) * PRICE["input_per_million"]
    oc = (out_tokens / 1_000_000) * PRICE["output_per_million"]
    return ic, oc, ic + oc

def build_messages(query: str, facts: list):
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
    def __init__(self, adapter, experiment: str):
        self.adapter = adapter
        self.experiment = experiment
        mlflow.set_experiment(experiment)

    def run(self, query: str, facts: list, temperature: float = 0.4):
        msgs = build_messages(query, facts)
        res = self.adapter.complete(msgs, temperature=temperature)
        in_t, out_t = res["in_tokens"], res["out_tokens"]
        ic, oc, tc = _cost(in_t, out_t)

        with mlflow.start_run() as run:
            mlflow.log_param("model", getattr(self.adapter, "model", "unknown"))
            mlflow.log_param("temperature", temperature)
            mlflow.log_param("query", query)

            mlflow.log_metric("input_tokens", in_t)
            mlflow.log_metric("output_tokens", out_t)
            mlflow.log_metric("latency_s", res["latency"])
            mlflow.log_metric("input_cost_usd", ic)
            mlflow.log_metric("output_cost_usd", oc)
            mlflow.log_metric("total_cost_usd", tc)

            # keep prompts and output for auditing or analysis
            mlflow.log_text(msgs[0]["content"], "system_prompt.txt")
            mlflow.log_text(msgs[-1]["content"], "user_prompt.txt")
            mlflow.log_text(res["text"], "assistant_response.txt")

            run_id = run.info.run_id

        return {
            "text": res["text"],
            "run_id": run_id,
            "cost_total_usd": tc,
        }
