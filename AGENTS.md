# Repository Guidelines

## Project Purpose & Priorities

This is a learning-oriented project for experiencing and understanding an end-to-end roleplay model post-training workflow. It is not intended to become a production training platform or a general-purpose model infrastructure project.

Two characteristics are equally important:

- **Simple and fast**: prefer the smallest viable datasets, one primary configuration, minimal dependencies and abstractions, and short feedback loops. Avoid production-grade machinery unless it is required for the current learning objective.
- **Complete pipeline**: preserve the full path from business goals, persona and data preparation through Base evaluation, SFT, GRPO, unified evaluation, and retrospective. Every core stage should produce a minimal inspectable artifact.

When these priorities appear to conflict, reduce the scale and complexity of each stage rather than removing a core stage. Optimize for a small, understandable, reproducible learning loop—not for feature breadth, throughput, or premature extensibility.

## Project Structure & Module Organization

This is a Python 3.10+ package using the `src` layout.

- `src/roleplay/`: application code. `persona.py` validates and renders persona data; `datagen.py` builds training/evaluation datasets; `inference.py` runs batch inference; `chat.py` provides interactive chat.
- `tests/`: `unittest` suites corresponding to the main modules.
- `data/`: JSON and JSONL persona, style, training, evaluation, and generated-output artifacts.
- `docs/`: project plans, execution guides, run records, issue archives, and retrospectives.

Keep reusable logic in `src/roleplay/`, CLI parsing in each module's `main()`, and deterministic API behavior covered with fake clients in `tests/`.

## Build, Test, and Development Commands

Create and activate a virtual environment, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Useful commands:

```bash
python -m unittest discover -s tests -v
roleplay-chat --help
roleplay-datagen --help
roleplay-inference --help
```

The first command runs the complete test suite. The remaining commands inspect the installed console interfaces. Chat and inference expect an OpenAI-compatible server (default `http://127.0.0.1:8080/v1`).

## Coding Style & Naming Conventions

Follow standard PEP 8 conventions: four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and `UPPER_CASE` constants. Add type hints to public functions and keep filesystem operations based on `pathlib.Path`. Preserve UTF-8 and use `ensure_ascii=False` for user-facing Chinese JSON. No formatter or linter is configured, so keep imports grouped and changes consistent with nearby code.

## Testing Guidelines

Tests use the standard-library `unittest` framework. Name files `test_<module>.py`, test classes by behavior (for example, `PersonaValidationTests`), and methods `test_<expected_behavior>`. Mock external API calls; tests must not require network access or real credentials. Add regression coverage for validation, retries, and atomic output behavior when changing data generation or inference.

## Commit & Pull Request Guidelines

History follows Conventional Commit prefixes such as `feat:`, `fix:`, `refactor:`, and `chore:`; concise English or Chinese subjects are accepted. Keep each commit focused. Pull requests should explain the behavior change, list tests run, link relevant issues, and call out changed data formats or CLI flags. Include sample command output when CLI behavior changes.

## Security & Configuration

Never commit API keys. Supply data-generation credentials through `DEEPSEEK_API_KEY` or the documented CLI option. Avoid committing large generated model artifacts; review JSONL outputs for sensitive conversation data before sharing.
