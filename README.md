# LLM_Summerization_adapter
A small demo project that builds a summary from a list of technical facts using Gemini. it logs input, output, prompts, tokens, cost, and latency ,and model parameters to mlflow and writes a short text report. 

## Project structure
llm_adapter_project/
│
├── main.py # main logic
├── run.py # small entry script
│
├── llm_adapter.py # llm interface + gemini adapter
├── pipeline.py # summarization pipeline class
├── utils.py # helpers: build prompt + write report
│
├── facts.json # sample data for summarization
├── requirements.txt # minimal dependencies
└── README.md


---

## Idea

the pipeline sends a user query + list of data points to the gemini model and gets back a short summary.  
everything is (query, model response, token counts, cost, latencya and model paramters) is logged to mlflow.  
Text file `llm_report.txt` is created for quick review.

the layout follows simple separation of concerns:

| layer | file | purpose |
|-------|------|----------|
| infrastructure | `llm_adapter.py` | handles model calls |
| application | `pipeline.py` | runs the summarization flow |
| helpers | `utils.py` | builds prompts + reports |
| interface | `main.py`, `run.py` | entry layer |
| data | `facts.json` | example input |

---

## how to run

1. install dependencies  
   ```bash
   pip install -r requirements.txt

## set your key, choose the model
export GEMINI_API_KEY="your_key"
export GEMINI_MODEL_NAME="gemini-2.5-flash"

## run the app

python run.py

which will load the facts, run the summarization, log metrics to mlflow and eventually create llm_report.txt
