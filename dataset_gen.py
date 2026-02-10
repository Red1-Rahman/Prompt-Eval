import json
from typing import List, Dict
from utils import GroqClient


class DatasetGenerator:
    """Generate test cases for prompt evaluation using Groq API"""
    
    def __init__(self, groq_client: GroqClient):
        self.client = groq_client
    
    def generate_test_cases(self, prompt: str, use_case_description: str, num_cases: int = 15) -> List[Dict]:
        """
        Generate diverse test cases for evaluating a prompt
        
        Args:
            prompt: The prompt template to evaluate
            use_case_description: Description of the use case (e.g., "email classifier")
            num_cases: Number of test cases to generate (default: 15)
        
        Returns:
            List of test case dictionaries with 'input' and 'expected_criteria'
        """
        generation_prompt = f"""You are a test case generator for LLM prompt evaluation.

Given this prompt template:
{prompt}

Use case: {use_case_description}

Generate {num_cases} diverse test cases that will thoroughly evaluate this prompt's performance.

For each test case, provide:
1. "input": The actual input text/query to test
2. "expected_criteria": What makes a good response (e.g., "should classify as urgent", "should extract 3 dates", "should be professional tone")
3. "difficulty": easy, medium, or hard
4. "category": A category label for organizing results

Make test cases diverse:
- Include edge cases (empty input, very long input, ambiguous cases)
- Cover different difficulty levels
- Test various aspects of the prompt's requirements
- Include both expected successes and challenging scenarios

Return ONLY valid JSON in this exact format:
{{
  "test_cases": [
    {{
      "input": "test input here",
      "expected_criteria": "description of what good output looks like",
      "difficulty": "easy",
      "category": "basic"
    }}
  ]
}}"""
        
        try:
            response = self.client.call(generation_prompt, temperature=0.8, max_tokens=2048, json_mode=True)
            data = json.loads(response)
            
            if "test_cases" in data:
                return data["test_cases"]
            else:
                # Fallback if structure is different
                return [data] if isinstance(data, dict) else data
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response was: {response}")
            return self._generate_fallback_cases(use_case_description, num_cases)
        except Exception as e:
            print(f"Error generating test cases: {e}")
            return self._generate_fallback_cases(use_case_description, num_cases)
    
    def _generate_fallback_cases(self, use_case: str, num_cases: int) -> List[Dict]:
        """Generate basic fallback test cases if API fails"""
        return [
            {
                "input": f"Test input {i+1} for {use_case}",
                "expected_criteria": "Should produce relevant and coherent output",
                "difficulty": "medium",
                "category": "general"
            }
            for i in range(min(num_cases, 5))
        ]
    
    def generate_comparison_cases(self, use_case_description: str, num_cases: int = 10) -> List[Dict]:
        """
        Generate test cases specifically designed for comparing multiple prompt versions
        Returns consistent test cases that can be reused across prompt variants
        """
        generation_prompt = f"""Generate {num_cases} consistent test cases for comparing different prompt versions.

Use case: {use_case_description}

These test cases will be used to evaluate multiple prompt variants side-by-side.
Make them:
- Representative of real-world usage
- Challenging enough to differentiate prompt quality
- Consistent in difficulty distribution

Return ONLY valid JSON:
{{
  "test_cases": [
    {{
      "input": "test input",
      "expected_criteria": "evaluation criteria",
      "difficulty": "easy|medium|hard",
      "category": "category name"
    }}
  ]
}}"""
        
        try:
            response = self.client.call(generation_prompt, temperature=0.7, max_tokens=2048, json_mode=True)
            data = json.loads(response)
            return data.get("test_cases", [])
        except:
            return self._generate_fallback_cases(use_case_description, num_cases)

