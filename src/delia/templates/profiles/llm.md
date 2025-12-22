# LLM Development Profile

Load this profile for: LLM applications, prompt engineering, fine-tuning, RAG, agents.

## Project Structure

```
project/
├── src/
│   ├── prompts/
│   │   ├── templates.py
│   │   └── few_shot.py
│   ├── chains/
│   │   ├── rag.py
│   │   └── agents.py
│   ├── embeddings/
│   │   └── vectorstore.py
│   ├── eval/
│   │   └── metrics.py
│   └── utils/
├── prompts/                 # Prompt templates
├── data/
│   ├── documents/
│   └── evaluations/
└── configs/
```

## Prompt Engineering

```python
from string import Template

# System prompts
SYSTEM_PROMPT = """You are a helpful assistant specialized in {domain}.

Rules:
- Be concise and accurate
- Cite sources when available
- Say "I don't know" if uncertain
- Never make up information"""

# Few-shot template
FEW_SHOT_TEMPLATE = """
Examples:
{examples}

Now respond to:
Input: {input}
Output:"""

def format_examples(examples: list[dict]) -> str:
    return "\n".join(
        f"Input: {ex['input']}\nOutput: {ex['output']}"
        for ex in examples
    )
```

## Structured Output

```python
from pydantic import BaseModel, Field
from typing import Literal

class ExtractedInfo(BaseModel):
    """Structured extraction schema."""
    name: str = Field(description="Person's full name")
    age: int | None = Field(description="Age if mentioned")
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1)

# With instructor or similar
import instructor
from openai import OpenAI

client = instructor.patch(OpenAI())

def extract_info(text: str) -> ExtractedInfo:
    return client.chat.completions.create(
        model="gpt-4",
        response_model=ExtractedInfo,
        messages=[
            {"role": "system", "content": "Extract information from text."},
            {"role": "user", "content": text},
        ],
    )
```

## RAG Pipeline

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def create_vectorstore(documents: list[str], persist_dir: str):
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Create embeddings and store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def query_with_context(query: str, vectorstore, llm, k: int = 4) -> str:
    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Generate response
    prompt = f"""Context:
{context}

Question: {query}

Answer based only on the context above:"""

    return llm.invoke(prompt)
```

## Fine-tuning Preparation

```python
import json

def prepare_training_data(examples: list[dict], output_path: str):
    """Prepare JSONL for fine-tuning."""
    with open(output_path, "w") as f:
        for ex in examples:
            entry = {
                "messages": [
                    {"role": "system", "content": ex.get("system", "You are a helpful assistant.")},
                    {"role": "user", "content": ex["input"]},
                    {"role": "assistant", "content": ex["output"]},
                ]
            }
            f.write(json.dumps(entry) + "\n")


# LoRA fine-tuning config
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
```

## Evaluation

```python
from typing import Callable

def evaluate_responses(
    test_cases: list[dict],
    generate_fn: Callable[[str], str],
    judge_fn: Callable[[str, str, str], float],
) -> dict:
    """Evaluate LLM responses."""
    results = []

    for case in test_cases:
        response = generate_fn(case["input"])
        score = judge_fn(case["input"], response, case.get("reference", ""))
        results.append({
            "input": case["input"],
            "response": response,
            "reference": case.get("reference"),
            "score": score,
        })

    return {
        "mean_score": sum(r["score"] for r in results) / len(results),
        "results": results,
    }
```

## Agent Patterns

```python
from typing import Callable

class Tool:
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def run(self, input: str) -> str:
        return self.func(input)


class Agent:
    def __init__(self, llm, tools: list[Tool], max_iterations: int = 5):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations

    def run(self, query: str) -> str:
        # ReAct loop
        for i in range(self.max_iterations):
            # Think
            thought = self._think(query, history)

            # Check if done
            if "FINAL ANSWER" in thought:
                return self._extract_answer(thought)

            # Act
            tool_name, tool_input = self._parse_action(thought)
            observation = self.tools[tool_name].run(tool_input)

            # Update history
            history.append({"thought": thought, "observation": observation})

        return "Max iterations reached"
```

## Best Practices

```
ALWAYS:
- Use structured outputs when possible
- Implement retry logic with exponential backoff
- Cache embeddings and responses
- Log prompts and responses for debugging
- Validate outputs against schemas

AVOID:
- Hardcoding prompts (use templates)
- Ignoring token limits
- Skipping evaluation
- Trusting LLM outputs blindly
```

## Token Management

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def truncate_to_limit(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])
```

