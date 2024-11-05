from typing import Any, Callable, List, Optional, Dict
from dataclasses import dataclass, field
import json
from enum import Enum
import re

class ValidationStrategy(Enum):
    STRICT = "strict"      # Most rigorous checking
    MODERATE = "moderate"  # Balanced approach
    RELAXED = "relaxed"    # More lenient checking

class HallucinationCategory(Enum):
    FACTUAL_ERROR = "factual_error"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    CONTEXT_DEVIATION = "context_deviation"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CONTRADICTORY_STATEMENT = "contradictory_statement"

@dataclass
class ValidationResult:
    is_hallucination: bool
    categories: List[HallucinationCategory] = field(default_factory=list)
    details: str = ""

@dataclass
class LLMResponse:
    content: str
    original_prompt: str
    context: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _validation_history: List[ValidationResult] = field(default_factory=list)

    def _clean_text(self, text: str) -> str:
        """Remove special characters and normalize text"""
        text = re.sub(r'[^\w\s.,?!]', '', text)
        return ' '.join(text.split()).lower()

    def _create_strict_validation_prompt(self) -> str:
        """Creates a highly rigorous validation prompt"""
        return f"""
        Objective: Perform an extremely thorough and strict analysis to detect any potential hallucinations.
        
        Original Prompt: {self.original_prompt}
        AI Response: {self.content}
        Context (if provided): {self.context}
        
        Perform a rigorous fact-checking analysis following these strict criteria:
        1. Every single statement must be verifiable with high confidence
        2. Any uncertain or unverifiable claim should be flagged
        3. Check for:
           - Exact factual accuracy of every detail
           - Complete logical consistency
           - Perfect alignment with the given context
           - Presence of any speculative content
           - Any embellishments or extrapolations
        
        Even minor inconsistencies or unverifiable details should be treated as hallucinations.
        
        Provide analysis in JSON format:
        {{
            "contains_hallucination": true/false,
            "categories": ["factual_error", "logical_inconsistency", "context_deviation", "unsupported_claim", "contradictory_statement"],
            "reasoning": "Detailed explanation of findings"
        }}
        """

    def _create_moderate_validation_prompt(self) -> str:
        """Creates a balanced validation prompt"""
        return f"""
        Objective: Evaluate the response for significant hallucinations while allowing for minor variations.
        
        Original Prompt: {self.original_prompt}
        AI Response: {self.content}
        Context (if provided): {self.context}
        
        Perform a balanced analysis focusing on:
        1. Core factual accuracy of main claims
        2. General logical consistency
        3. Reasonable alignment with context
        4. Presence of major unsupported claims
        
        Minor imprecisions or reasonable extrapolations are acceptable.
        
        Provide analysis in JSON format:
        {{
            "contains_hallucination": true/false,
            "categories": ["factual_error", "logical_inconsistency", "context_deviation", "unsupported_claim", "contradictory_statement"],
            "reasoning": "Detailed explanation of findings"
        }}
        """

    def _create_relaxed_validation_prompt(self) -> str:
        """Creates a more lenient validation prompt"""
        return f"""
        Objective: Check for major hallucinations while allowing for reasonable interpretation and elaboration.
        
        Original Prompt: {self.original_prompt}
        AI Response: {self.content}
        Context (if provided): {self.context}
        
        Perform a basic analysis focusing only on:
        1. Major factual errors
        2. Significant logical flaws
        3. Substantial deviations from context
        
        Allow for:
        - Reasonable interpretations
        - Minor factual imprecisions
        - Creative elaborations within reason
        - Generalizations
        
        Only flag serious violations or major inconsistencies.
        
        Provide analysis in JSON format:
        {{
            "contains_hallucination": true/false,
            "categories": ["factual_error", "logical_inconsistency", "context_deviation", "unsupported_claim", "contradictory_statement"],
            "reasoning": "Detailed explanation of findings"
        }}
        """

    def _get_strategy_specific_prompts(self, strategy: ValidationStrategy) -> List[str]:
        """Get validation prompts based on the selected strategy"""
        if strategy == ValidationStrategy.STRICT:
            return [
                self._create_strict_validation_prompt(),
                f"""Perform an extremely detailed fact-check of this statement, flagging any uncertainty:
                    Statement: {self.content}""",
                f"""Analyze this response for any logical inconsistencies or unverifiable claims, no matter how minor:
                    Response: {self.content}"""
            ]
        elif strategy == ValidationStrategy.MODERATE:
            return [
                self._create_moderate_validation_prompt(),
                f"""Check if this statement contains any significant factual errors:
                    Statement: {self.content}""",
                f"""Evaluate if this response is reasonably consistent with the question and context:
                    Question: {self.original_prompt}
                    Response: {self.content}"""
            ]
        else:  # RELAXED
            return [
                self._create_relaxed_validation_prompt(),
                f"""Check if this statement contains any major, obvious errors:
                    Statement: {self.content}""",
                f"""Is this response broadly appropriate for the given question?
                    Question: {self.original_prompt}
                    Response: {self.content}"""
            ]

    def _parse_validation_response(self, response: str) -> ValidationResult:
        """Parse and validate the LLM's validation response"""
        try:
            if isinstance(response, str):
                response_dict = json.loads(response)
            else:
                response_dict = response

            if not all(k in response_dict for k in ['contains_hallucination', 'categories', 'reasoning']):
                raise ValueError("Missing required fields in validation response")

            categories = []
            for cat in response_dict['categories']:
                try:
                    categories.append(HallucinationCategory(cat))
                except ValueError:
                    continue

            return ValidationResult(
                is_hallucination=response_dict['contains_hallucination'],
                categories=categories,
                details=response_dict['reasoning']
            )

        except json.JSONDecodeError:
            # Fallback parsing
            contains_hallucination = "hallucination" in response.lower() or "incorrect" in response.lower()
            return ValidationResult(
                is_hallucination=contains_hallucination,
                categories=[HallucinationCategory.UNSUPPORTED_CLAIM],
                details="Failed to parse validation response"
            )

    def check_hallucination(
        self, 
        llm_function: Callable,
        strategy: ValidationStrategy = ValidationStrategy.STRICT
    ) -> bool:
        """
        Enhanced hallucination detection with strategy-specific validation.
        """
        # Get strategy-specific prompts
        validation_prompts = self._get_strategy_specific_prompts(strategy)
        
        # Perform validation checks
        validation_results = []
        for prompt in validation_prompts:
            response = llm_function(prompt)
            validation_results.append(self._parse_validation_response(response))
        
        # Store results in history
        self._validation_history.extend(validation_results)
        
        # Apply strategy-specific evaluation
        if strategy == ValidationStrategy.STRICT:
            # Any detection is considered a hallucination
            return any(result.is_hallucination for result in validation_results)
            
        elif strategy == ValidationStrategy.MODERATE:
            # Majority voting with consideration of category severity
            hallucination_count = sum(1 for result in validation_results if result.is_hallucination)
            return hallucination_count > len(validation_results) // 2
            
        else:  # RELAXED
            # Only flag if all checks detect serious hallucinations
            serious_categories = {
                HallucinationCategory.FACTUAL_ERROR,
                HallucinationCategory.CONTRADICTORY_STATEMENT
            }
            return all(
                result.is_hallucination and 
                any(cat in serious_categories for cat in result.categories)
                for result in validation_results
            )

    def get_validation_details(self) -> Dict[str, Any]:
        """Get detailed validation results"""
        if not self._validation_history:
            return {"error": "No validation performed yet"}
            
        return {
            "total_checks": len(self._validation_history),
            "hallucination_detections": sum(1 for r in self._validation_history if r.is_hallucination),
            "categories_found": list(set(cat for r in self._validation_history for cat in r.categories)),
            "detailed_results": [
                {
                    "is_hallucination": r.is_hallucination,
                    "categories": [cat.value for cat in r.categories],
                    "details": r.details
                }
                for r in self._validation_history
            ]
        }

# Add this at the end of the file:
class HallucinationChecker:
    """
    Main class for hallucination detection functionality
    """
    def __init__(self, llm_function: Callable):
        self.checker = HallucinationChecker(llm_function)
        self.ValidationStrategy = ValidationStrategy  # Make enum accessible through the class

    def check(
        self,
        content: str,
        prompt: str,
        context: Any = None,
        strategy: ValidationStrategy = ValidationStrategy.STRICT
    ) -> bool:
        """
        Main method to check for hallucinations
        """
        return self.checker.check_response(
            content=content,
            original_prompt=prompt,
            context=context,
            strategy=strategy
        )

    def get_details(self) -> Dict[str, Any]:
        """
        Get detailed analysis of all checks
        """
        return self.checker.get_history_analysis()