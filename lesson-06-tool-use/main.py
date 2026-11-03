"""Tool calling with Groq — Lesson 4.6."""

import ast
import json
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from groq import Groq

from groq.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from typing import cast

load_dotenv()

client = Groq()

MODEL = "openai/gpt-oss-120b"

ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def get_current_time(timezone: str) -> dict:
    """Return the current time in an IANA timezone."""

    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)

        return {
            "timezone": timezone,
            "datetime": now.isoformat(),
            "formatted": now.strftime(
                "%A, %B %d, %Y at %I:%M %p %Z"
            ),
        }

    except Exception:
        return {
            "error": f"Invalid timezone: {timezone}"
        }


def safe_eval(node):
    """Evaluate a restricted mathematical AST node."""

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(
                f"Unsupported constant: {node.value!r}"
            )

        return node.value

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPS:
            raise ValueError(
                f"Unsupported operator: {operator_type.__name__}"
            )

        left = safe_eval(node.left)
        right = safe_eval(node.right)

        operation = ALLOWED_OPS[operator_type]

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPS:
            raise ValueError(
                f"Unsupported unary operator: "
                f"{operator_type.__name__}"
            )

        operand = safe_eval(node.operand)

        operation = ALLOWED_OPS[operator_type]

        return operation(operand)

    raise ValueError(
        f"Unsupported expression: {type(node).__name__}"
    )


def calculate(expression: str) -> dict:
    """Safely evaluate a simple arithmetic expression."""

    try:
        tree = ast.parse(
            expression,
            mode="eval",
        )

        result = safe_eval(tree.body)

        return {
            "expression": expression,
            "result": result,
        }

    except Exception as error:
        return {
            "error": f"Could not evaluate expression: {error}"
        }


def get_word_count(text: str) -> dict:
    """Count words and characters in a piece of text."""

    words = text.split()

    return {
        "word_count": len(words),
        "character_count": len(text),
        "character_count_no_spaces": len(
            text.replace(" ", "")
        ),
    }


TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current date and time for a location using an IANA "
                "timezone identifier. Use this when the user asks for the "
                "current time, date, or local time in a city or timezone. "
                "Pass timezone values such as 'Asia/Tokyo', 'Asia/Kolkata', "
                "'Europe/London', or 'America/New_York'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": (
                            "An IANA timezone identifier, for example "
                            "'Asia/Tokyo' or 'Asia/Kolkata'."
                        ),
                    }
                },
                "required": ["timezone"],
                "additionalProperties": False,
            },
        },
    },
]

TOOLS.append(
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Safely evaluate a simple arithmetic expression. Use this "
                "when the user asks for an exact mathematical calculation "
                "involving +, -, *, /, **, %, or parentheses. "
                "Pass only the mathematical expression, without prose."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "The arithmetic expression to evaluate, for example "
                            "'47 * 83 + 129' or '(10 + 5) / 3'."
                        ),
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    }
)

TOOLS.append(
    {
        "type": "function",
        "function": {
            "name": "get_word_count",
            "description": (
                "Count words and characters in an exact piece of text. "
                "Use this when the user asks how many words or characters "
                "are in a sentence, paragraph, or other supplied text. "
                "Pass only the text that should be counted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "The exact text to count. Do not include the user's "
                            "surrounding question or quotation marks unless they "
                            "are part of the text itself."
                        ),
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    }
)


TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_word_count": get_word_count,
}


def run_conversation_with_tools(
    user_message: str,
    verbose: bool = True,
) -> dict:
    """Run a conversation that may involve tool calls."""

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": user_message,
        }
    ]

    iterations = 0
    tool_calls_made = []
    max_iterations = 10

    for _ in range(max_iterations):
        iterations += 1

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )

        assistant_message = response.choices[0].message

        # No tool calls means the model has produced its final answer.
        if not assistant_message.tool_calls:
            return {
                "answer": assistant_message.content or "",
                "iterations": iterations,
                "tool_calls": tool_calls_made,
            }

        # Preserve the assistant's tool-call request in conversation history.
        assistant_message_dict = cast(
            ChatCompletionMessageParam,
            assistant_message.model_dump(
                exclude_none=True,
            ),
        )

        messages.append(assistant_message_dict)

        # One model response can request multiple tools.
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name

            # Tool arguments arrive from the API as a JSON string.
            arguments = json.loads(
                tool_call.function.arguments
            )

            if verbose:
                print(
                    f"  → Model calls: "
                    f"{function_name}({arguments})"
                )

            # Map the tool name chosen by the LLM
            # to the actual Python function.
            if function_name in TOOL_FUNCTIONS:
                function = TOOL_FUNCTIONS[
                    function_name
                ]

                result = function(**arguments)

            else:
                result = {
                    "error": (
                        f"Unknown function: "
                        f"{function_name}"
                    )
                }

            if verbose:
                print(
                    f"  ← Tool returns: {result}"
                )

            # Keep metadata for our own debugging / experiment output.
            tool_calls_made.append(
                {
                    "name": function_name,
                    "arguments": arguments,
                    "result": result,
                }
            )

            # Send this specific tool result back to the model.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

        # No return here.
        # The loop runs again so the model can read the tool results
        # and either request more tools or produce its final answer.

    raise RuntimeError(
        f"Exceeded {max_iterations} tool call iterations"
    )


def main() -> None:
    """Run tool-calling experiments."""

    queries = [
        "What time is it in Tokyo right now?",
        "Calculate 47 * 83 + 129",
        (
            "How many words are in the sentence: "
            "'The quick brown fox jumps over the lazy dog'?"
        ),
        "What is 2+2 and what time is it in Bengaluru?",
        "Who wrote Hamlet?",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        result = run_conversation_with_tools(
            user_message=query,
            verbose=True,
        )

        print(f"\nFinal answer: {result['answer']}")
        print(f"Iterations: {result['iterations']}")
        print(
            f"Tool calls: "
            f"{len(result['tool_calls'])}"
        )


if __name__ == "__main__":
    main()

