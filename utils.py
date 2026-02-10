import os
import json
from typing import Dict, List, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class GroqClient:
    """Groq API client wrapper for easy LLM calls"""
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            # Let Groq automatically load from GROQ_API_KEY environment variable
            self.client = Groq()
        self.model = "llama-3.1-8b-instant"
    
    def call(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024, json_mode: bool = False) -> str:
        """Make a call to Groq API"""
        try:
            response_format = {"type": "json_object"} if json_mode else None
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stream=False,
                response_format=response_format
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def call_with_system(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, 
                        max_tokens: int = 1024, json_mode: bool = False) -> str:
        """Make a call with both system and user messages"""
        try:
            response_format = {"type": "json_object"} if json_mode else None
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stream=False,
                response_format=response_format
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"


def save_results(data: Dict, filename: str):
    """Save results to JSON file"""
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    return filepath


def load_results(filename: str) -> Dict:
    """Load results from JSON file"""
    filepath = os.path.join("results", filename)
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_stats(scores: List[float]) -> Dict:
    """Calculate statistics from scores"""
    if not scores:
        return {
            "average": 0,
            "min": 0,
            "max": 0,
            "count": 0
        }
    
    return {
        "average": round(sum(scores) / len(scores), 2),
        "min": min(scores),
        "max": max(scores),
        "count": len(scores),
        "pass_rate": round(len([s for s in scores if s >= 7]) / len(scores) * 100, 1)
    }

