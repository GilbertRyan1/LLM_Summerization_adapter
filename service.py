# service.py
from typing import List, Dict, Any, Tuple
import datetime
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import load_text

PRICE = {
    "input_per_million": 0.10,
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

def _ts_to_str(timestamp_ms):
    if isinstance(timestamp_ms, (int, float)):
        dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000.0)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return "N/A"

def write_mlflow_text_report(experiment_name: str, output_filename: str = "llm_run_report.txt"):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        print(f"experiment not found: {experiment_name}")
        return

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        print("no runs found")
        return

    run = runs[0]
    run_id = run.info.run_id
    base = f"runs:/{run_id}/"

    metrics = run.data.metrics
    params = run.data.params

    system_txt = load_text(base + "system_prompt.txt")
    user_txt = load_text(base + "user_prompt.txt")
    out_txt = load_text(base + "assistant_response.txt")

    template = """
LLM run report

run id:        {run_id}
experiment id: {experiment_id}
status:        {status}
artifact uri:  {artifact_uri}
start time:    {start_time}

model:         {model_name}
temperature:   {temperature}
query:         {query}

total tokens:  {total_tokens}
input tokens:  {input_tokens} ({input_cost})
output tokens: {output_tokens} ({output_cost})
total cost:    {total_cost}

[system prompt]
{system_txt}

[user prompt]
{user_txt}

[model output]
{out_txt}
"""
    report = template.format(
        run_id=run_id,
        experiment_id=run.info.experiment_id,
        status=run.info.status,
        artifact_uri=run.info.artifact_uri,
        start_time=_ts_to_str(run.info.start_time),
        model_name=params.get("model", "N/A"),
        temperature=params.get("temperature", "N/A"),
        query=params.get("query", "N/A"),
        total_tokens=metrics.get("input_tokens", 0) + metrics.get("output_tokens", 0),
        input_tokens=metrics.get("input_tokens", 0),
        output_tokens=metrics.get("output_tokens", 0),
        input_cost=f"${metrics.get('input_cost_usd', 0):.6f}",
        output_cost=f"${metrics.get('output_cost_usd', 0):.6f}",
        total_cost=f"${metrics.get('total_cost_usd', 0):.6f}",
        system_txt=system_txt,
        user_txt=user_txt,
        out_txt=out_txt,
    )

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(report.strip())

    print(f"report saved: {output_filename}")

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
