import json
import re
from typing import Dict, List, Callable
from textblob import TextBlob
from utils import GroqClient


class CodeBasedGraders:
    """Collection of code-based grading functions (100% free)"""
    
    @staticmethod
    def length_validator(response: str, min_length: int = 10, max_length: int = 5000) -> Dict:
        """Validate response length"""
        length = len(response)
        
        if length < min_length:
            return {
                "score": 3,
                "reason": f"Response too short ({length} chars, minimum {min_length})",
                "passed": False
            }
        elif length > max_length:
            return {
                "score": 5,
                "reason": f"Response too long ({length} chars, maximum {max_length})",
                "passed": False
            }
        else:
            return {
                "score": 10,
                "reason": f"Length appropriate ({length} chars)",
                "passed": True
            }
    
    @staticmethod
    def json_validator(response: str) -> Dict:
        """Validate if response is valid JSON"""
        try:
            json.loads(response)
            return {
                "score": 10,
                "reason": "Valid JSON format",
                "passed": True
            }
        except json.JSONDecodeError as e:
            return {
                "score": 0,
                "reason": f"Invalid JSON: {str(e)}",
                "passed": False
            }
    
    @staticmethod
    def keyword_checker(response: str, required_keywords: List[str], case_sensitive: bool = False) -> Dict:
        """Check if response contains required keywords"""
        if not case_sensitive:
            response = response.lower()
            required_keywords = [k.lower() for k in required_keywords]
        
        found_keywords = [kw for kw in required_keywords if kw in response]
        missing_keywords = [kw for kw in required_keywords if kw not in response]
        
        score = int((len(found_keywords) / len(required_keywords)) * 10) if required_keywords else 10
        
        return {
            "score": score,
            "reason": f"Found {len(found_keywords)}/{len(required_keywords)} keywords. Missing: {missing_keywords}" if missing_keywords else "All keywords present",
            "passed": len(missing_keywords) == 0,
            "details": {
                "found": found_keywords,
                "missing": missing_keywords
            }
        }
    
    @staticmethod
    def regex_matcher(response: str, pattern: str, should_match: bool = True) -> Dict:
        """Check if response matches a regex pattern"""
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        
        if should_match:
            if match:
                return {
                    "score": 10,
                    "reason": f"Pattern matched: {pattern}",
                    "passed": True
                }
            else:
                return {
                    "score": 0,
                    "reason": f"Pattern not found: {pattern}",
                    "passed": False
                }
        else:
            if not match:
                return {
                    "score": 10,
                    "reason": f"Pattern correctly absent: {pattern}",
                    "passed": True
                }
            else:
                return {
                    "score": 0,
                    "reason": f"Unwanted pattern found: {pattern}",
                    "passed": False
                }
    
    @staticmethod
    def sentiment_analyzer(response: str, expected_sentiment: str = "neutral") -> Dict:
        """Analyze sentiment using TextBlob (free library)"""
        try:
            blob = TextBlob(response)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            # Classify sentiment
            if polarity > 0.1:
                detected = "positive"
            elif polarity < -0.1:
                detected = "negative"
            else:
                detected = "neutral"
            
            matches = detected == expected_sentiment.lower()
            score = 10 if matches else max(0, 10 - abs(polarity * 10))
            
            return {
                "score": int(score),
                "reason": f"Detected {detected} sentiment (polarity: {polarity:.2f}), expected {expected_sentiment}",
                "passed": matches,
                "details": {
                    "polarity": round(polarity, 3),
                    "detected": detected,
                    "expected": expected_sentiment
                }
            }
        except Exception as e:
            return {
                "score": 5,
                "reason": f"Error analyzing sentiment: {str(e)}",
                "passed": False
            }
    
    @staticmethod
    def format_validator(response: str, format_type: str) -> Dict:
        """Validate response format (email, url, phone, etc.)"""
        patterns = {
            "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            "url": r'https?://[^\s]+',
            "phone": r'(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}',
            "date": r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}',
            "number": r'^-?\d+\.?\d*$'
        }
        
        if format_type not in patterns:
            return {
                "score": 5,
                "reason": f"Unknown format type: {format_type}",
                "passed": False
            }
        
        match = re.search(patterns[format_type], response.strip())
        
        return {
            "score": 10 if match else 0,
            "reason": f"Valid {format_type} format" if match else f"Invalid {format_type} format",
            "passed": bool(match)
        }


class ModelBasedGrader:
    """Use Groq API for intelligent grading (free)"""
    
    def __init__(self, groq_client: GroqClient):
        self.client = groq_client
    
    def grade_response(self, test_case: Dict, response: str, custom_criteria: str = "") -> Dict:
        """
        Grade a response using LLM
        
        Args:
            test_case: Dict with 'input' and 'expected_criteria'
            response: The LLM response to grade
            custom_criteria: Optional additional grading criteria
        """
        # Check if response is an error first
        if response.startswith("Error:"):
            return {
                "score": 0,
                "reason": "API error prevented response generation",
                "strengths": [],
                "weaknesses": ["API connection issue"],
                "is_technical_error": True
            }
        
        grading_prompt = f"""Score this LLM response on a scale of 1-10.

INPUT: {test_case.get('input', 'N/A')}

EXPECTED CRITERIA: {test_case.get('expected_criteria', 'General quality')}

{"ADDITIONAL CRITERIA: " + custom_criteria if custom_criteria else ""}

RESPONSE TO GRADE:
{response}

Evaluate based on:
1. Accuracy - Does it address the input correctly?
2. Completeness - Does it meet the expected criteria?
3. Format compliance - Is it properly formatted?
4. Clarity - Is it clear and coherent?

Return ONLY a JSON object with this exact structure:
{{
  "score": <number 1-10>,
  "reason": "<brief explanation of the score>",
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"]
}}"""
        
        try:
            response_text = self.client.call(grading_prompt, temperature=0.3, max_tokens=500, json_mode=True)
            
            # Check if grading call itself failed
            if response_text.startswith("Error:"):
                return {
                    "score": 0,
                    "reason": "Grading service temporarily unavailable",
                    "strengths": [],
                    "weaknesses": ["API connection issue"],
                    "is_technical_error": True
                }
            
            result = json.loads(response_text)
            
            # Ensure score is valid
            if "score" not in result:
                result["score"] = 5
            result["score"] = max(1, min(10, int(result["score"])))
            
            # Ensure reason exists
            if "reason" not in result:
                result["reason"] = "No specific reason provided"
            
            # Ensure arrays exist
            if "strengths" not in result:
                result["strengths"] = []
            if "weaknesses" not in result:
                result["weaknesses"] = []
            
            result["is_technical_error"] = False
            return result
            
        except json.JSONDecodeError:
            return {
                "score": 0,
                "reason": "Unable to parse grading response",
                "strengths": [],
                "weaknesses": ["API response format issue"],
                "is_technical_error": True
            }
        except Exception as e:
            return {
                "score": 0,
                "reason": f"Grading unavailable: {str(e)[:50]}",
                "strengths": [],
                "weaknesses": ["API connection issue"],
                "is_technical_error": True
            }
    
    def batch_grade(self, results: List[Dict], custom_criteria: str = "") -> List[Dict]:
        """Grade multiple responses"""
        graded_results = []
        
        for result in results:
            grade = self.grade_response(
                result.get("test_case", {}),
                result.get("response", ""),
                custom_criteria
            )
            
            graded_result = result.copy()
            graded_result["grade"] = grade
            graded_results.append(graded_result)
        
        return graded_results


class CustomGrader:
    """Allow users to define custom grading functions"""
    
    def __init__(self):
        self.custom_validators: Dict[str, Callable] = {}
    
    def register_validator(self, name: str, validator_func: Callable):
        """Register a custom validation function"""
        self.custom_validators[name] = validator_func
    
    def run_validator(self, name: str, response: str, **kwargs) -> Dict:
        """Run a registered validator"""
        if name not in self.custom_validators:
            return {
                "score": 0,
                "reason": f"Validator '{name}' not found",
                "passed": False
            }
        
        try:
            return self.custom_validators[name](response, **kwargs)
        except Exception as e:
            return {
                "score": 0,
                "reason": f"Error running validator: {str(e)}",
                "passed": False
            }


# Pre-built validator templates
def create_word_count_validator(min_words: int, max_words: int) -> Callable:
    """Template: Validate word count"""
    def validator(response: str) -> Dict:
        words = len(response.split())
        if min_words <= words <= max_words:
            return {
                "score": 10,
                "reason": f"Word count in range ({words} words)",
                "passed": True
            }
        else:
            score = max(0, 10 - abs(words - (min_words + max_words) // 2) // 10)
            return {
                "score": score,
                "reason": f"Word count out of range ({words} words, expected {min_words}-{max_words})",
                "passed": False
            }
    return validator


def create_structure_validator(required_sections: List[str]) -> Callable:
    """Template: Validate document structure (headers, sections)"""
    def validator(response: str) -> Dict:
        found = [s for s in required_sections if s.lower() in response.lower()]
        missing = [s for s in required_sections if s not in found]
        
        score = int((len(found) / len(required_sections)) * 10)
        
        return {
            "score": score,
            "reason": f"Found {len(found)}/{len(required_sections)} required sections",
            "passed": len(missing) == 0,
            "details": {"found": found, "missing": missing}
        }
    return validator

