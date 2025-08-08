# Summary
Short version:

* **OpenAI:** you set `reasoning.effort: "low" | "medium" | "high"`. It changes how many *internal reasoning tokens* the model is allowed to burn before writing the final answer. Default is **medium**. ([OpenAI Platform][1], [Microsoft Learn][2])
* **Anthropic (Claude “thinking”):** you explicitly give a **token budget** for thinking, e.g. `thinking: {type: "enabled", budget_tokens: 16000}`. Those tokens count toward your `max_tokens` and there’s a minimum (\~1,024). ([Anthropic][3], [AWS Documentation][4], [Promptfoo][5])
* **Google Gemini (reasoning/thinking models):** you set an integer `thinkingBudget` (0 disables it). Docs show ranges up to \~24,576 tokens, and recommend raising it for tougher tasks. ([Google AI for Developers][6], [gemini-api.apidog.io][7])

# Do these map 1:1?

No official, vendor-blessed conversion exists from OpenAI’s **low/medium/high** to exact Anthropic/Gemini token counts. OpenAI deliberately doesn’t publish fixed token numbers per level; they only say higher effort uses more internal tokens. You *can* see the **actual** `reasoning_tokens` used after the call in usage metadata, but that’s post-hoc—not a knob. ([OpenAI Platform][1], [Microsoft Learn][2])

# A pragmatic equivalence (rule-of-thumb)

If you need a starting point to make the three vendors “feel” similar in latency/cost/quality, these rough budgets usually line up well in practice:

| OpenAI effort |                        Anthropic `budget_tokens` |       Gemini `thinkingBudget` |
| ------------- | -----------------------------------------------: | ----------------------------: |
| **low**       |                                    \~1,000–2,000 |                 \~1,000–2,000 |
| **medium**    |                                    \~4,000–8,000 |                 \~4,000–8,000 |
| **high**      | \~12,000–16,000 (go to 24k for gnarly math/code) | \~12,000–16,000 (up to \~24k) |

> Why these numbers? Anthropic/Gemini expose a direct token cap, while OpenAI exposes a coarse **effort** setting. Since OpenAI doesn’t publish fixed counts, the only honest approach is to pick budgets that match your latency/quality targets and then tune by measuring outcomes (accuracy vs. time/cost) and, for OpenAI, checking the **observed** `reasoning_tokens` in responses to see what “low/medium/high” used on *your* tasks. ([Google AI for Developers][6], [Anthropic][3], [Microsoft Learn][2])

# Tuning tips

* **OpenAI:** Start at `medium`. If answers are shallow/incorrect, try `high`. Track `completion_tokens_details.reasoning_tokens` to understand the real cost. ([Microsoft Learn][2])
* **Anthropic:** Enable thinking and set `budget_tokens` below your `max_tokens` (≥1,024). Increase in steps (2k → 8k → 16k) until quality plateaus. ([Anthropic][3], [AWS Documentation][4])
* **Gemini:** Use `thinkingBudget`. 0 for speed, a few thousand for normal tasks, five figures for hard reasoning. ([Google AI for Developers][6], [gemini-api.apidog.io][7])

If you want, tell me the task type(s) you’re running (math proofs, long-code refactors, policy analysis, etc.) and I’ll give you a tighter set of defaults per vendor.


# Anthropic
In the request, use thinking = ...

The response has a new block type (thinking). Convert that to an empty text block with meta={'user_sees': <thinking summary>}

```
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    messages=[{
        "role": "user",
        "content": "Are there an infinite number of prime numbers such that n mod 4 == 3?"
    }]
)

# The response will contain summarized thinking blocks and text blocks
for block in response.content:
    if block.type == "thinking":
        print(f"\nThinking summary: {block.thinking}")
    elif block.type == "text":
        print(f"\nResponse: {block.text}")
```

# OpenAI

## GPT-5
```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5",
    input="How much gold would it take to coat the Statue of Liberty in a 1mm layer?",
    reasoning={
        "effort": "minimal"
    }
)

print(response)
```

## Other models
For all other openai models, don't use reasoning={...}


# Google
Generating content with thinking

Initiating a request with a thinking model is similar to any other content generation request. The key difference lies in specifying one of the models with thinking support in the model field, as demonstrated in the following text generation example:
Python
JavaScript
Go
REST

from google import genai

client = genai.Client()
prompt = "Explain the concept of Occam's Razor and provide a simple, everyday example."
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=prompt
)

print(response.text)

Thinking budgets

The thinkingBudget parameter guides the model on the number of thinking tokens to use when generating a response. A higher token count generally allows for more detailed reasoning, which can be beneficial for tackling more complex tasks. If latency is more important, use a lower budget or disable thinking by setting thinkingBudget to 0. Setting the thinkingBudget to -1 turns on dynamic thinking, meaning the model will adjust the budget based on the complexity of the request.

The thinkingBudget is only supported in Gemini 2.5 Flash, 2.5 Pro, and 2.5 Flash-Lite. Depending on the prompt, the model might overflow or underflow the token budget.

The following are thinkingBudget configuration details for each model type.
Model 	Default setting
(Thinking budget is not set) 	Range 	Disable thinking 	Turn on dynamic thinking
2.5 Pro 	Dynamic thinking: Model decides when and how much to think 	128 to 32768 	N/A: Cannot disable thinking 	thinkingBudget = -1
2.5 Flash 	Dynamic thinking: Model decides when and how much to think 	0 to 24576 	thinkingBudget = 0 	thinkingBudget = -1
2.5 Flash Lite 	Model does not think 	512 to 24576 	thinkingBudget = 0 	thinkingBudget = -1
Python
JavaScript
Go
REST

from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Provide a list of 3 famous physicists and their key contributions",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=1024)
        # Turn off thinking:
        # thinking_config=types.ThinkingConfig(thinking_budget=0)
        # Turn on dynamic thinking:
        # thinking_config=types.ThinkingConfig(thinking_budget=-1)
    ),
)

print(response.text)

Thought summaries

Thought summaries are synthesized versions of the model's raw thoughts and offer insights into the model's internal reasoning process. Note that thinking budgets apply to the model's raw thoughts and not to thought summaries.

You can enable thought summaries by setting includeThoughts to true in your request configuration. You can then access the summary by iterating through the response parameter's parts, and checking the thought boolean.

Here's an example demonstrating how to enable and retrieve thought summaries without streaming, which returns a single, final thought summary with the response:
Python
JavaScript
Go

from google import genai
from google.genai import types

client = genai.Client()
prompt = "What is the sum of the first 50 prime numbers?"
response = client.models.generate_content(
  model="gemini-2.5-pro",
  contents=prompt,
  config=types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
      include_thoughts=True
    )
  )
)

for part in response.candidates[0].content.parts:
  if not part.text:
    continue
  if part.thought:
    print("Thought summary:")
    print(part.text)
    print()
  else:
    print("Answer:")
    print(part.text)
    print()

And here is an example using thinking with streaming, which returns rolling, incremental summaries during generation:
Python
JavaScript
Go

from google import genai
from google.genai import types

client = genai.Client()

prompt = """
Alice, Bob, and Carol each live in a different house on the same street: red, green, and blue.
The person who lives in the red house owns a cat.
Bob does not live in the green house.
Carol owns a dog.
The green house is to the left of the red house.
Alice does not own a cat.
Who lives in each house, and what pet do they own?
"""

thoughts = ""
answer = ""

for chunk in client.models.generate_content_stream(
    model="gemini-2.5-pro",
    contents=prompt,
    config=types.GenerateContentConfig(
      thinking_config=types.ThinkingConfig(
        include_thoughts=True
      )
    )
):
  for part in chunk.candidates[0].content.parts:
    if not part.text:
      continue
    elif part.thought:
      if not thoughts:
        print("Thoughts summary:")
      print(part.text)
      thoughts += part.text
    else:
      if not answer:
        print("Answer:")
      print(part.text)
      answer += part.text

Thought signatures

Because standard Gemini API text and content generation calls are stateless, when using thinking in multi-turn interactions (such as chat), the model doesn't have access to thought context from previous turns.

You can maintain thought context using thought signatures, which are encrypted representations of the model's internal thought process. The model returns thought signatures in the response object when thinking and function calling are enabled. To ensure the model maintains context across multiple turns of a conversation, you must provide the thought signatures back to the model in the subsequent requests.

You will receive thought signatures when:

    Thinking is enabled and thoughts are generated.
    The request includes function declarations.

Note: Thought signatures are only available when you're using function calling, specifically, your request must include function declarations.

You can find an example of thinking with function calls on the Function calling page.

Other usage limitations to consider with function calling include:

    Signatures are returned from the model within other parts in the response, for example function calling or text parts. Return the entire response with all parts back to the model in subsequent turns.
    Don't concatenate parts with signatures together.
    Don't merge one part with a signature with another part without a signature.
