# localrouter
This is an automatically generated overview of the current workspace.

## Files

- uv.lock
- pyproject.toml
- README.md
- .gitignore
- LLM.md

## Python
This project uses uv to manage dependencies. The default python points to the local venv. Use `uv add <package>` to install a package.

## Recent Updates

### Reasoning/Thinking Support (Latest)
Added support for reasoning budgets across OpenAI, Anthropic, and Google Gemini models:

- **ReasoningConfig** class for configuring reasoning/thinking behavior
- Supports three configuration styles:
  - `effort`: "minimal"/"low"/"medium"/"high" (OpenAI-style)
  - `budget_tokens`: int (Anthropic/Gemini-style explicit token count)
  - `dynamic`: bool (Gemini-style, let model decide)
- Automatic conversion between formats when switching providers
- Graceful handling of models that don't support reasoning (config is ignored)

Example usage:
```python
from localrouter import get_response, ReasoningConfig

# Using effort levels
response = await get_response(
    model="gemini-2.5-pro",
    messages=messages,
    reasoning=ReasoningConfig(effort="high")
)

# Using explicit token budget
response = await get_response(
    model="claude-sonnet-4-20250514",  # When available
    messages=messages,
    reasoning=ReasoningConfig(budget_tokens=8000)
)

# Using dict config (backward compatible)
response = await get_response(
    model="gpt-5",  # When available
    messages=messages,
    reasoning={"effort": "medium"}
)
```

## Updating this file

This file should serve as an onboarding guide for you in the future. Keep it up-to-date with info about:
- the purpose of the project
- the state of the code base
- any other relevant information
