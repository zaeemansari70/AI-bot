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
