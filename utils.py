import datetime
from mlflow.tracking import MlflowClient
from mlflow.artifacts import load_text

def build_llm_messages(query: str, facts: list):
    blocks = []
    for item in facts:
        topic = item.get("Topic", "N/A")
        content = item.get("Content", "N/A")
        blocks.append(f"[topic: {topic}]\n{content}")
    doc_block = "\n\n---\n\n".join(blocks)

    prompt = (
        "You summarize technical content.\n\n"
        f"User query:\n{query}\n\n"
        "Facts:\n"
        f"{doc_block}\n\n"
        "Write one short paragraph that answers the query. "
        "Do not add extra knowledge, headings, or lists."
    ).strip()

    return [
        {"role": "system", "content": "Summarize technical content for the user."},
        {"role": "user", "content": prompt},
    ]


def _ts_to_str(timestamp_ms):
    if isinstance(timestamp_ms, (int, float)):
        dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000.0)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return "N/A"


def create_text_log_report(experiment_name: str, output_filename: str):
    client = MlflowClient()

    try:
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
            model_name=params.get("model_name", "N/A"),
            temperature=params.get("temperature", "N/A"),
            start_time=_ts_to_str(run.info.start_time),
            total_tokens=metrics.get("total_tokens", 0),
            input_tokens=metrics.get("input_tokens", 0),
            output_tokens=metrics.get("output_tokens", 0),
            input_cost=f"${metrics.get('input_cost_usd', 0):.6f}",
            output_cost=f"${metrics.get('output_cost_usd', 0):.6f}",
            total_cost=f"${metrics.get('total_cost_usd', 0):.6f}",
            query=params.get("query", "N/A"),
            system_txt=system_txt,
            user_txt=user_txt,
            out_txt=out_txt,
        )

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(report.strip())

        print(f"report saved: {output_filename}")
    except Exception as e:
        print(f"error while creating report: {e}")
