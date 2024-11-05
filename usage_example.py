import ollama
from ollama import Options
from halucheck import (
    HallucinationChecker,
    ValidationStrategy
)

def ask_llm(query: str) -> str:

    user_prompt = f"""
    What is the capital of this County:
    {query}
    """
    
    response = ollama.chat(
        model='gemma2:2b-instruct-q8_0',
        options=Options(temperature=0.0),
        messages=[
            {
                'role': 'system',
                'content': "You tell capitals of the cities based on the Country name.",
            },
            {
                'role': 'user',
                'content': user_prompt,
            },
        ]
    )
    
    answer = response['message']['content'].strip()

    return answer

prompt = 'france'

response = ask_llm(prompt)

detector = HallucinationChecker(ask_llm)

# Use different strategies
strict_check = detector.check(
    content=response,
    prompt=prompt,
    strategy=ValidationStrategy.STRICT
)

print(strict_check)

moderate_check = detector.check(
    content=response,
    prompt=prompt,
    strategy=ValidationStrategy.MODERATE
)

print(moderate_check)

relaxed_check = detector.check(
    content=response,
    prompt=prompt,
    strategy=ValidationStrategy.RELAXED
)

print(relaxed_check)