# Architecture Notes

This system implements a judge-driven, self-improving CSV analysis agent.

## Components
- User CLI: Interactive prompt for questions.
- Worker LLM: Generates tool calls and pandas code.
- Tools Layer: Safe pandas execution and dataset utilities.
- Judge LLM: Validates worker outputs and proposes corrections.
- Vector DB: Stores corrected patterns for cross-session learning.

## Data Flow
1) User question arrives at the CLI.
2) Similar past mistakes are retrieved from FAISS and injected into the system prompt.
3) Worker LLM generates tool calls (search/validate + pandas_exec).
4) Tools execute pandas code in a sandbox and return a result.
5) Judge LLM evaluates correctness using dataset context + tool output.
6) If wrong, corrected code is executed and the fix is queued.
7) Final answer is rendered from the latest tool output.
8) On exit, queued mistakes are persisted into FAISS.

## Persistence Strategy
- Each stored entry includes question, error description, corrected code, and wrong code.
- Retrieval is semantic (embedding search), then injected as "Learned rules".
- Numeric values in stored patterns are scrubbed to avoid leakage.

## Safety
- Pandas execution is validated via AST allow-listing.
- Tool outputs are the sole source of numeric answers.

## Production Notes
- Move FAISS to a managed store if running at scale.
- Add query/answer logging and metrics for monitoring.
- Implement a rollback mechanism for bad corrections.

## Design Document (Detailed)

### Architecture Overview
**System Diagram**
- Diagram: `docs/architecture/diagram.svg`
- Components: User CLI, Worker LLM, Tools Layer, Judge LLM, Dataset Card, Vector DB (FAISS), Answer Renderer.

**Data Flow**
1) User question arrives in the CLI.
2) Similar mistakes are retrieved from FAISS and injected into the system prompt.
3) Worker LLM produces tool calls and pandas code.
4) Tools execute code and return results.
5) Judge LLM validates output using dataset context and tool results.
6) If incorrect, corrected code is executed and queued for persistence.
7) Final response is rendered strictly from the latest tool output.
8) On exit, queued mistakes are stored in FAISS.

**Agentic Loop**
- The loop is deterministic: question → tool calls → tool execution → judge validation → correction (if needed) → answer rendering.
- The worker never returns final numeric answers directly; outputs are derived from tools.
- The judge is always invoked, ensuring a consistent validation pass for every query.

**Improvement Storage + Retrieval**
- Storage: FAISS local store (`faiss_db`) containing documents with question, error description, corrected code, and wrong code.
- Retrieval: semantic similarity search on the current question; matched items are injected into the system prompt as “Learned rules”.
- Justification: FAISS provides lightweight, local persistence without additional infra; it is fast and easy to version/control.

### Self-Improvement Mechanism
**Trigger**
- Any judge evaluation that marks `error=yes` and returns a `corrected_code` triggers a queued improvement.

**Representation**
- Each improvement is stored as a compact text document:
  - QUESTION, ERROR, RULE (if any), CORRECTED_CODE, WRONG_CODE.
  - Numeric values are scrubbed to avoid overfitting to specific outputs.

**Storage**
- Stored locally in FAISS (`faiss_db`) at shutdown for durability.
- This aligns with the take-home scope and avoids introducing external dependencies.

**Application in New Sessions**
- On each query, similar mistakes are retrieved and injected into the worker prompt.
- This biases the worker toward prior corrections and reduces repeat failures.

### Code Execution Strategy
**Approach**
- Safe pandas execution with `exec()` and an AST allow-list (`safety.py`).
- Only a restricted set of builtins is exposed.
- The code must assign to `result` to be considered valid.

**Safety/Sandboxing**
- AST validation blocks imports, file I/O, and dunder access.
- Execution context is limited to `df` and `pd`.

**Trade-offs**
- Pros: fast, simple, and deterministic; no external sandbox required.
- Cons: still runs inside the process; for production, containerized execution is safer.

### Evaluation Strategy
**Effectiveness**
- Judge output provides a structured error signal.
- Improvements are considered effective if similar future questions no longer trigger corrections.
- Manual spot checks: compare baseline vs. post-learning answers.

**Preventing Bad Improvements**
- Only store corrections when corrected code executes successfully.
- In production: add rollback/blacklist mechanisms and offline validation suites.

### Production Considerations
**Changes Needed**
- Replace local FAISS with a managed vector store or database.
- Add Re-ranking.
- Add structured logging, tracing, and metrics for tool calls and judge results.
- Harden sandboxing (containerized execution per request).

**Scalability**
- Separate the tool runner and model inference into independent services.
- Cache dataset cards and frequent query results.

**Security**
- Strict sandboxing for generated code.
- Secret management for API keys.
- Audit logs for user inputs and tool execution.

### AI Tool Usage
**Tools Used**
- Groq LLMs via `langchain_groq` for worker and judge.
- FAISS + sentence-transformers for persistent memory.

**Human vs. Generated Work**
- I mostly came up with the architecture, the AI helped me write alot of code, I've used Claude, Grok and ChatGPT, a little but of gemini, mostly for generating code of the architecture I made myself. Validated the architecture from the LLM's. I made sure the prompts going in are okay or not. Judged the code produced by LLM everytime and told it to correct if any mistake it made. Had to give them some documentation too, as they might hallucinate if they don't have the updated docs.

