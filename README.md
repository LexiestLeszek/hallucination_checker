# LLM Hallucination Checker

A robust and flexible library for detecting hallucinations in Large Language Model (LLM) responses.

## Features

- Multiple validation strategies: Strict, Moderate, and Relaxed
- Detailed categorization of hallucinations
- Customizable LLM function integration
- Comprehensive validation history and reporting

## Installation

```bash
pip install llm-hallucination-checker
```

## Quick Start

```python
from llm_hallucination_detector import HallucinationChecker, ValidationStrategy

# Define your LLM function
def my_llm_function(prompt):
    # Your LLM API call here
    pass

# Initialize the checker
checker = HallucinationChecker(my_llm_function)

# Check for hallucinations
content = "The sky is green."
prompt = "Describe the sky."
is_hallucination = checker.check(content, prompt, strategy=ValidationStrategy.STRICT)

print(f"Contains hallucination: {is_hallucination}")
```

## Detailed Usage

### Initialization

```python
from llm_hallucination_detector import HallucinationChecker, ValidationStrategy, LLMResponse

checker = HallucinationChecker(my_llm_function)
```

### Checking for Hallucinations

```python
# Basic check
is_hallucination = checker.check(content, prompt)

# Check with context
is_hallucination = checker.check(content, prompt, context="Some additional context")

# Using different validation strategies
is_hallucination = checker.check(content, prompt, strategy=ValidationStrategy.MODERATE)
```

### Advanced Usage with LLMResponse

```python
response = LLMResponse(content="The sky is green.", original_prompt="Describe the sky.")
is_hallucination = response.check_hallucination(my_llm_function, ValidationStrategy.STRICT)

# Get detailed validation results
details = response.get_validation_details()
print(details)
```

## Validation Strategies

- `STRICT`: Most rigorous checking. Flags even minor inconsistencies.
- `MODERATE`: Balanced approach. Allows for minor imprecisions.
- `RELAXED`: More lenient checking. Only flags major inconsistencies.

## Hallucination Categories

- `FACTUAL_ERROR`
- `LOGICAL_INCONSISTENCY`
- `CONTEXT_DEVIATION`
- `UNSUPPORTED_CLAIM`
- `CONTRADICTORY_STATEMENT`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
