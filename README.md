# CSV Bot - Self-Improving Data Analysis Agent

This project implements a CSV analysis chatbot with a persistent learning loop.
It answers natural language questions against a CSV, validates answers with a
judge model, and stores corrections in a vector database for future sessions.

## Features
- Natural language to pandas analysis via tool calling.
- Judge-in-the-loop validation on every query.
- Persistent improvement storage (FAISS).
- Metadata responses (rows/columns/schema/preview).
- Safe pandas execution with AST validation.

## Requirements
- Python 3.11+
- A Groq API key

## Setup
1) Create a `.env` file:
```
GROQ_API_KEY=your_key_here
CSV_PATH=FUN_company_pl_actuals_dataset.csv
```

2) Install dependencies:
```
pip install -r requirements.txt
```

3) Run the bot:
```
python main.py
```

## Docker
Build:
```
docker build -t csv-bot .
```

Run (mount the CSV and .env):
```
docker run --rm -it \
  -v "$PWD/FUN_company_pl_actuals_dataset.csv:/app/FUN_company_pl_actuals_dataset.csv" \
  -v "$PWD/.env:/app/.env" \
  csv-bot
```

## Project Structure
- `main.py`: Main agent loop, tool handling, and judge flow.
- `tools.py`: Dataset access and safe pandas execution.
- `judge.py`: Judge model prompt and validation logic.
- `vectordb.py`: FAISS persistence for learned corrections.
- `docs/architecture/`: Architecture diagram and notes.

## Demo Checklist
1) Ask a trick question (Product E) and confirm the error response.
2) Ask a standard query and confirm tool execution.
3) Ask a hard query (YoY/growth) and confirm judge correction if needed.
4) Exit and verify FAISS entries were stored.
