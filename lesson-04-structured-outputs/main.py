"""Structured outputs with Pydantic — Lesson 4.4."""

import json

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ValidationError, field_validator


load_dotenv()

client = Groq()

MODEL = "openai/gpt-oss-120b"


class ProductReturn(BaseModel):
    """One validated product return extracted from an email."""

    product_name: str
    price: float
    order_number: str
    purchase_date: str
    issue: str
    customer_email: str
    customer_phone: str

    @field_validator("price", mode="before")
    @classmethod
    def coerce_price(cls, value):
        """Clean common currency formatting before price validation."""

        if isinstance(value, str):
            value = (
                value
                .replace("$", "")
                .replace(",", "")
                .strip()
            )

        return value


class ExtractionResult(BaseModel):
    """Validated collection of product returns."""

    returns: list[ProductReturn]




def extract_returns(
    email: str,
    error_feedback: str = "",
) -> ExtractionResult:
    """Extract and validate product returns from one customer email."""

    schema_hint = json.dumps(
        ExtractionResult.model_json_schema(),
        indent=2,
    )

    system_prompt = f"""
    You are a customer service data analyst.

    Extract product return requests from customer emails.

    Return a JSON object matching this exact schema:

    {schema_hint}

    Rules:
    - Only include products the customer wants to RETURN.
    - Do not include products they are only asking about.
    - Use ISO 8601 format for dates: YYYY-MM-DD.
    - Price must be a number, not a string.
    - If a required field is missing, do not include that product in returns.
    - If no complete return can be extracted, return {{"returns": []}}.
    - Do not include markdown or explanatory text.
    {error_feedback}
    """

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": email,
            },
        ],
        response_format={
            "type": "json_object",
        },
    )

    raw_output = response.choices[0].message.content

    if raw_output is None:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    parsed_json = json.loads(raw_output)

    return ExtractionResult.model_validate(
        parsed_json
    )


def extract_returns_with_retry(
    email: str,
    max_retries: int = 3,
) -> tuple[ExtractionResult, int]:
    """Extract returns with retries on parse or validation failure."""

    error_feedback = ""
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            result = extract_returns(
                email=email,
                error_feedback=error_feedback,
            )

            return result, attempt

        except (ValidationError, json.JSONDecodeError) as error:
            last_error = error

            error_feedback = (
                "\n\nPrevious attempt failed. "
                f"Error: {error}\n"
                "Correct the response and return valid JSON "
                "matching the schema exactly."
            )

    raise RuntimeError(
        f"Extraction failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


EMAIL_1 = """
Hi, I need to return the Logitech MX Master 3 wireless mouse
($99.99, order ORD-2026-8842) purchased on July 15th.

The scroll wheel stopped working after 3 days.

My email is jane.doe@example.com,
phone +1-555-0123.
"""


EMAIL_2 = """
Hello support team,

I need to return two items from my recent order
ORD-2026-9915 placed on August 3rd:

1. The Anker USB-C hub ($34.99) - it's completely dead,
no lights, no functionality.

2. The Belkin wireless charger ($49.99) - it charges very slowly,
maybe 1% per 10 minutes.

Please email me at mike.wilson@example.com with return instructions.
Phone: +1-555-0287
"""


EMAIL_3 = """
Hi there,

I got the Sony WH-1000XM5 headphones ($399.99)
on July 22nd, order #ORD-2026-7734.

They arrived with a broken hinge.
I want to return them.

Also, can you tell me if you have the Bose QC45 in stock?
Just asking, not ordering yet.

Thanks,
Sarah Chen
sarah.chen@example.com
+1-555-0344
"""


MALFORMED_EMAIL = """
Return the item please. It broke.
"""


def main() -> None:
    """Run structured extraction against several customer emails."""

    emails = [
        ("Email 1 — single return", EMAIL_1),
        ("Email 2 — multiple returns", EMAIL_2),
        ("Email 3 — return plus inquiry", EMAIL_3),
        ("Malformed email", MALFORMED_EMAIL),
    ]

    for label, email in emails:
        print("\n" + "=" * 60)
        print(label)
        print("=" * 60)

        try:
            result, retries = extract_returns_with_retry(email)

            print(f"Success: YES")
            print(f"Retries used: {retries}")

            print(
                json.dumps(
                    result.model_dump(),
                    indent=2,
                )
            )

            if result.returns:
                first_price = result.returns[0].price

                print(
                    f"First price Python type: "
                    f"{type(first_price).__name__}"
                )

        except RuntimeError as error:
            print("Success: NO")
            print(f"Error: {error}")


if __name__ == "__main__":
    main()