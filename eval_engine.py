import time
from typing import Dict, List, Optional
from datetime import datetime
from utils import GroqClient, save_results, calculate_stats
from dataset_gen import DatasetGenerator
from graders import ModelBasedGrader, CodeBasedGraders


class EvaluationEngine:
    """Main orchestrator for prompt evaluation"""
    
    def __init__(self, groq_client: GroqClient):
        self.client = groq_client
        self.dataset_gen = DatasetGenerator(groq_client)
        self.model_grader = ModelBasedGrader(groq_client)
        self.code_graders = CodeBasedGraders()
    
    def run_evaluation(self, prompt: str, test_cases: List[Dict], 
                      use_model_grading: bool = True,
                      code_graders: Optional[List[str]] = None,
                      temperature: float = 0.7) -> Dict:
        """
        Run complete evaluation of a prompt against test cases
        
        Args:
            prompt: The prompt to evaluate
            test_cases: List of test case dictionaries
            use_model_grading: Whether to use LLM-based grading
            code_graders: List of code-based grader names to apply
            temperature: Temperature for prompt execution
        
        Returns:
            Complete evaluation results
        """
        results = []
        start_time = time.time()
        
        print(f"Running evaluation on {len(test_cases)} test cases...")
        
        for idx, test_case in enumerate(test_cases, 1):
            print(f"  Processing test case {idx}/{len(test_cases)}...", end="\r")
            
            # Execute prompt with test input
            full_prompt = f"{prompt}\n\n{test_case['input']}"
            response = self.client.call(full_prompt, temperature=temperature, max_tokens=1024)
            
            result = {
                "test_case": test_case,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            
            # Apply code-based grading
            if code_graders:
                result["code_grades"] = {}
                for grader_name in code_graders:
                    if hasattr(self.code_graders, grader_name):
                        grader_func = getattr(self.code_graders, grader_name)
                        result["code_grades"][grader_name] = grader_func(response)
            
            # Apply model-based grading
            if use_model_grading:
                result["model_grade"] = self.model_grader.grade_response(test_case, response)
            
            results.append(result)
        
        print(f"\nCompleted {len(test_cases)} test cases in {time.time() - start_time:.2f}s")
        
        # Calculate overall statistics (exclude technical errors)
        scores = [r["model_grade"]["score"] for r in results 
                 if "model_grade" in r and not r["model_grade"].get("is_technical_error", False)]
        
        if not scores:
            # All requests failed - provide feedback
            stats = {
                "average": 0,
                "min": 0,
                "max": 0,
                "count": 0,
                "pass_rate": 0,
                "error": "All evaluations failed. Please check your API key."
            }
        else:
            stats = calculate_stats(scores)
            # Track failed evaluations
            failed = len(results) - len(scores)
            if failed > 0:
                stats["failed_evaluations"] = failed
        
        return {
            "prompt": prompt,
            "results": results,
            "stats": stats,
            "metadata": {
                "total_cases": len(test_cases),
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": round(time.time() - start_time, 2)
            }
        }
    
    def compare_prompts(self, prompts: List[Dict[str, str]], test_cases: List[Dict],
                       use_model_grading: bool = True) -> Dict:
        """
        Compare multiple prompt versions on same test cases
        
        Args:
            prompts: List of dicts with 'name' and 'prompt' keys
            test_cases: Shared test cases for comparison
            use_model_grading: Whether to use LLM grading
        
        Returns:
            Comparison results with side-by-side scores
        """
        comparison_results = {
            "prompts": prompts,
            "test_cases": test_cases,
            "evaluations": {},
            "comparison": {
                "winner": None,
                "summary": {}
            }
        }
        
        print(f"\nComparing {len(prompts)} prompt versions...")
        
        # Run evaluation for each prompt
        for prompt_info in prompts:
            name = prompt_info["name"]
            prompt = prompt_info["prompt"]
            
            print(f"\n--- Evaluating: {name} ---")
            eval_result = self.run_evaluation(
                prompt, 
                test_cases, 
                use_model_grading=use_model_grading
            )
            
            comparison_results["evaluations"][name] = eval_result
            comparison_results["comparison"]["summary"][name] = eval_result["stats"]
        
        # Determine winner
        best_score = 0
        winner = None
        for name, stats in comparison_results["comparison"]["summary"].items():
            if stats["average"] > best_score:
                best_score = stats["average"]
                winner = name
        
        comparison_results["comparison"]["winner"] = winner
        
        # Generate improvement analysis
        if len(prompts) == 2:
            comparison_results["comparison"]["improvements"] = self._analyze_improvements(
                comparison_results["evaluations"][prompts[0]["name"]],
                comparison_results["evaluations"][prompts[1]["name"]],
                prompts[0]["name"],
                prompts[1]["name"]
            )
        
        return comparison_results
    
    def _analyze_improvements(self, eval1: Dict, eval2: Dict, name1: str, name2: str) -> Dict:
        """Analyze improvements/regressions between two evaluations"""
        improvements = []
        regressions = []
        
        for i, (r1, r2) in enumerate(zip(eval1["results"], eval2["results"])):
            if "model_grade" in r1 and "model_grade" in r2:
                score1 = r1["model_grade"]["score"]
                score2 = r2["model_grade"]["score"]
                diff = score2 - score1
                
                if diff > 0:
                    improvements.append({
                        "test_case_index": i,
                        "input": r1["test_case"]["input"][:100] + "...",
                        "improvement": diff,
                        "score_change": f"{score1} → {score2}"
                    })
                elif diff < 0:
                    regressions.append({
                        "test_case_index": i,
                        "input": r1["test_case"]["input"][:100] + "...",
                        "regression": abs(diff),
                        "score_change": f"{score1} → {score2}"
                    })
        
        return {
            "improvements": sorted(improvements, key=lambda x: x["improvement"], reverse=True)[:5],
            "regressions": sorted(regressions, key=lambda x: x["regression"], reverse=True)[:5],
            "net_change": eval2["stats"]["average"] - eval1["stats"]["average"]
        }
    
    def suggest_improvements(self, evaluation_results: Dict) -> List[str]:
        """
        Analyze evaluation results and suggest prompt improvements
        
        Args:
            evaluation_results: Results from run_evaluation
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        stats = evaluation_results["stats"]
        results = evaluation_results["results"]
        
        avg_score = stats["average"]
        scores = [r["model_grade"]["score"] for r in results if "model_grade" in r]
        
        # Low average score
        if avg_score < 5:
            suggestions.append("💡 Average score is low. Try adding examples (one-shot or few-shot prompting)")
            suggestions.append("💡 Consider breaking complex tasks into smaller steps (chain-of-thought)")
        
        # High variance in scores
        if scores:
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            if variance > 6:
                suggestions.append("💡 Scores vary widely. Lower temperature to 0.3 for more consistent outputs")
                suggestions.append("💡 Add explicit constraints or rules to reduce variability")
        
        # Format issues (check code_grades if available)
        format_violations = 0
        for result in results:
            if "code_grades" in result:
                for grade_name, grade in result["code_grades"].items():
                    if not grade.get("passed", True):
                        format_violations += 1
        
        if format_violations > len(results) * 0.3:
            suggestions.append("💡 Many format violations detected. Use XML tags or structured output format")
            suggestions.append("💡 Add explicit format instructions with examples")
        
        # Analyze common weaknesses (filter out technical errors)
        all_weaknesses = []
        technical_errors = 0
        for result in results:
            if "model_grade" in result:
                grade = result["model_grade"]
                # Check if this is a technical error
                if grade.get("is_technical_error", False):
                    technical_errors += 1
                elif "weaknesses" in grade:
                    # Only count real content weaknesses, not technical errors
                    for weakness in grade["weaknesses"]:
                        if weakness.lower() not in ["grading error occurred", "grading error", "api connection issue", "api response format issue"]:
                            all_weaknesses.append(weakness)
        
        # If many technical errors, add a note
        if technical_errors > len(results) * 0.3:
            suggestions.insert(0, "⚠️ Some responses could not be graded due to API issues. Check your API key and connection.")
        
        if all_weaknesses:
            weakness_counts = {}
            for weakness in all_weaknesses:
                weakness_lower = weakness.lower()
                weakness_counts[weakness_lower] = weakness_counts.get(weakness_lower, 0) + 1
            
            # Find most common weaknesses
            common_weaknesses = sorted(weakness_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            if common_weaknesses:
                suggestions.append(f"💡 Common issues detected: {', '.join([w[0] for w in common_weaknesses])}")
        
        # General best practices
        if not suggestions:
            suggestions.append("✅ Performance looks good! Consider testing edge cases")
            suggestions.append("✅ Try A/B testing with slight variations to optimize further")
        
        return suggestions
    
    def generate_report(self, evaluation_results: Dict, filename: Optional[str] = None) -> str:
        """
        Generate and save evaluation report
        
        Args:
            evaluation_results: Results from run_evaluation or compare_prompts
            filename: Optional filename for saving report
        
        Returns:
            Path to saved report
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_report_{timestamp}.json"
        
        filepath = save_results(evaluation_results, filename)
        return filepath

