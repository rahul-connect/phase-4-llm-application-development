from dotenv import load_dotenv
from groq import Groq
import json

load_dotenv()

client = Groq()


MODEL = "openai/gpt-oss-120b"

EMAIL_TEXT = """
Hi there,

I need to return the wireless mouse I bought last week. It's the Logitech MX Master 3
in black, cost me $99.99 on July 15th. Order number was ORD-2026-8842. The scroll wheel
stopped working after 3 days. My email is jane.doe@example.com and my phone is +1-555-0123.

Also, I wanted to ask about the mechanical keyboard I saw on your site - the Keychron K2
- is that in stock? Not ready to buy yet, just checking.

Thanks,
Jane Doe
"""


def extract_returns(
        email:str,
        prompt_style:str
) -> dict:
    if prompt_style == "naive":
        system = None

        user_content = (
            "Extract product return information "
            "from this email as JSON.\n\n"
            f"{email}"
        )

    elif prompt_style == "role_and_task":
         system = (
            "You are a customer service data analyst. "
            "Extract product return information from customer emails."
        )

         user_content = (
            "Return a JSON array of return items with these fields: "
            "product_name, price, order_number, purchase_date, issue, "
            "customer_email, customer_phone.\n\n"
            f"{email}"
        )

    elif prompt_style == "structured_with_delimiters":
        system = (
            "You are a customer service data analyst. "
            "Extract product return information from customer emails."
        )

        user_content = f"""
            # Input Email

            <email>
            {email}
            </email>

            # Output Format

            Return a JSON array.
            - If no products are being returned, return an empty JSON array [].

            Each object in the array must have exactly these fields:

            - product_name: string
            - price: number (example: 99.99, not "$99.99")
            - order_number: string
            - purchase_date: string in YYYY-MM-DD format
            - issue: short description
            - customer_email: string
            - customer_phone: string

            # Rules

            - Include only products the customer wants to return.
            - Do not include products the customer is only asking about.
            - If a field is not present in the email, use null.
            - Return only the JSON array.
            - Do not include markdown or any text before or after the JSON.
            """
    elif prompt_style == "few_shot":
        system = (
            "You are a customer service data analyst. "
            "Extract product return information from customer emails."
        )

        user_content = f"""
        # Examples

        ## Example 1

        Email:

        Hi,

        I bought an office chair for Rs.10000 on 2026-08-20.
        Order number ORD-2026-4521.

        The chair has a manufacturing defect and I would like to return it.

        My email is amit@example.com.

        I am also interested in buying a television if one is available in the same price range.

        Expected Output:

        [
        {{
            "product_name": "Office Chair",
            "price": 10000,
            "order_number": "ORD-2026-4521",
            "purchase_date": "2026-08-20",
            "issue": "Manufacturing defect",
            "customer_email": "amit@example.com",
            "customer_phone": null
        }}
        ]

        ## Example 2

        Email:

        Hi,

        I bought a JBL headphone on 2026-09-10 for Rs.4500.
        Order number ORD-2026-9124.

        The sound quality was poor, so I would like to return it.

        You can reach me at rahul@example.com or +91-9876543210.

        I am also interested in JBL high-end speakers with strong bass.

        Expected Output:

        [
        {{
            "product_name": "JBL Headphone",
            "price": 4500,
            "order_number": "ORD-2026-9124",
            "purchase_date": "2026-09-10",
            "issue": "Poor sound quality",
            "customer_email": "rahul@example.com",
            "customer_phone": "+91-9876543210"
        }}
        ]

        # Now process the following customer email

        <email>
        {email}
        </email>

        # Output Format

        Return a JSON array.

        Each object must contain exactly these fields:

        - product_name
        - price
        - order_number
        - purchase_date (convert to ISO format YYYY-MM-DD whenever possible)
        - issue
        - customer_email
        - customer_phone

        # Rules

        - Include only products being returned.
        - Ignore products the customer is only asking about.
        - Use null for missing fields.
        - Return only the JSON array.
        - Do not include markdown or explanatory text.
        """

    elif prompt_style == "chain_of_thought":
        system = (
            "You are a customer service data analyst. "
            "Carefully analyze customer emails and extract product return information."
        )

        user_content = f"""
        # Task

        Analyze the customer email using this process:

        1. Identify every product mentioned in the email.
        2. For each product, determine whether the customer wants to return it
        or is only asking about it.
        3. Ignore products that are not being returned.
        4. For each returned product, identify the required fields that are present.
        5. Use null for any required field that cannot be determined from the email.
        6. Normalize the purchase date to YYYY-MM-DD only when the full date can be
        determined from the email or trusted context.

        # Input Email

        <email>
        {email}
        </email>

        # Output Format

        Return a JSON array.

        Each object must contain exactly these fields:

        - product_name
        - price
        - order_number
        - purchase_date
        - issue
        - customer_email
        - customer_phone

        # Rules

        - Include only products the customer wants to return.
        - Do not include products the customer is only asking about.
        - price must be a number without a currency symbol.
        - Use null for missing or indeterminable fields.
        - Return only the final JSON array.
        - Do not include markdown, explanations, or reasoning in the final response.
        """

    else:
        raise ValueError(
            f"Unknown prompt style: {prompt_style}"
        )

    messages = []

    if system is not None:
        messages.append(
            {
                "role": "system",
                "content": system,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=messages,
    )

    usage = response.usage

    if usage is None:
        raise RuntimeError(
            "Groq response did not include token usage."
        )

    return {
        "style": prompt_style,
        "text": response.choices[0].message.content,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }





def main() -> None:
    """Run all prompt strategies and compare their outputs."""

    styles = [
        "naive",
        "role_and_task",
        "structured_with_delimiters",
        "few_shot",
        "chain_of_thought",
    ]

    results = [
        extract_returns(EMAIL_TEXT, style)
        for style in styles
    ]

    for result in results:
        print("\n" + "=" * 60)
        print(f"Strategy: {result['style']}")
        print(f"Completion tokens: {result['completion_tokens']}")
        print(f"Total tokens: {result['total_tokens']}")
        print("=" * 60)

        text = result["text"]

        print(text)

        try:
            parsed = json.loads(text)
            print("\nValid JSON: YES")
            print(f"Parsed Python type: {type(parsed).__name__}")

        except json.JSONDecodeError as exc:
            print("\nValid JSON: NO")
            print(f"JSON error: {exc}")


if __name__ == "__main__":
    main()
