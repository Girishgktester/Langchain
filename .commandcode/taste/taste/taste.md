# Taste
- Prefers a commit-and-push to `main` workflow after each batch of work, frequently requesting "push latest changes" with a meaningful commit message. Confidence: 0.9
- Prefers descriptive, meaningful commit messages that reflect the actual diff (e.g. "Add parallel and conditional runnable examples"), and explicitly asks for detailed/explanatory commit messages when work is more involved. Confidence: 0.9
- Works in a Windows environment using Jupyter notebooks (`.ipynb`) for LangChain learning; shell commands are executed through Windows `cmd` (not PowerShell), so avoid PowerShell-only syntax like `Select-Object`. Confidence: 0.8
- Prefers staging specific changed files (using `git add <specific paths>`) and combining stage + commit in one command rather than staging everything blindly. Confidence: 0.7
- Willing to knowingly publish potentially sensitive files to an unverified/destination repo when explicitly asked, after being informed of the risk (e.g. approving push of a profile PDF). Confidence: 0.6
- Reports errors very tersely (e.g. "fix this errror") without pasting the traceback or error text, expecting the agent to dig into the notebook/file and environment to find and diagnose the failure itself. Confidence: 0.6
- Develops and tests LangChain/RAG work against local Ollama models (`ChatOllama` with `qwen3:8b`, `nomic-embed-text` embeddings via `langchain-ollama` + Chroma) rather than hosted LLM APIs, and evaluates agents with RAGAS and DeepEval RAG metrics (e.g. AnswerRelevancy, Faithfulness, ContextualRecall) using the same local model as judge. Confidence: 0.8
- Expects new Python packages to be installed "matching my current config" — i.e. version-compatible with the existing environment, checking installed versions/constraints and avoiding downgrades or breakage of the existing stack before installing. Confidence: 0.7
- Works inside a project virtualenv (e.g. `myenv321`, Python 3.12) and expects installs and commands to target that interpreter rather than global Python. Confidence: 0.6
 hosted LLM APIs. Confidence: 0.7
