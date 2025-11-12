

from typing import List, Dict, Any
from llm_adapter import LLMInterface
from utils import build_llm_messages, create_text_log_report


class SummarizationPipeline:
    def __init__(
        self,
        adapter: LLMInterface,
        query: str,
        facts: List[Dict[str, Any]],
        experiment_name: str,
        report_file: str,
    ):
        self.adapter = adapter
        self.query = query
        self.facts = facts
        self.experiment_name = experiment_name
        self.report_file = report_file

    def run(self, temperature: float = 0.4) -> Dict[str, Any]:
        msgs = build_llm_messages(self.query, self.facts)
        out = self.adapter.generate_and_log_completion(
            messages=msgs,
            temperature=temperature,
            context={"query": self.query, "facts_sent": len(self.facts)},
        )
        create_text_log_report(self.experiment_name, self.report_file)
        return out
