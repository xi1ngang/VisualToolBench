#!/usr/bin/env python3
"""
Combined evaluation script that grades model responses and analyzes results.
Run LLM judge for multi-turn model response.
"""

import json
import argparse
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import math
import shutil
import dotenv
import openai

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class LLM_Judge:
    def __init__(self, client, model_name: str):
        self.model_name = model_name
        self.client = client

    def LLMasJudge(self, question, golden_answer, model_answer, rubric_criteria):
        prompt = f"""
        You are an expert evaluator tasked with judging whether a model's answer meets a specific rubric criterion.  
        You will be provided with:  
        - a question  
        - a golden (reference) answer  
        - a rubric criterion  
        - the model's answer  

        Your task is to decide if the model's answer **meets** or **does not meet** the given rubric criterion, referencing the golden answer only as needed.

        ### Inputs:
        **Question:** {question}  

        **Golden Answer:** {golden_answer}  

        **Rubric Criterion:** {rubric_criteria}  

        **Model Answer:** {model_answer}  

        ### Important Notes:
        - The model's answer does not need to be correct to meet the criterion if correctness is not required.  
        *Example:* If the rubric is "The model should show its reasoning process to answer the question," the answer can be incorrect but still meet the rubric if model's reasoning process is present.  
        - For writing style or presentation rubrics, apply leniency.  
        *Example:* If the rubric asks for conciseness, answers that are slightly longer than the golden answer but still reasonably length should be considered as meeting the rubric.
        - The model's answer may satisfy the rubric implicitly without explicitly mentioning the exact term. This should still be considered as meeting the criterion if model's answer is reasonable and makes sense.  
        *Example:* If the rubric is "The model should demonstrate understanding of photosynthesis," and the model states "Plants make their own food using sunlight," without explicitly mentioning the term "photosynthesis," it still meets the criterion.

        ### Output Format:
        Return your judgement in the following JSON format:
        {{
            "explanation": "Brief explanation of your judgement",
            "judge_result": "Met" or "Not Met"
        }}
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

def setup_client():
    """Setup OpenAI client with environment variables."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client

def process_chunk_judging(chunk_data, chunk_id, total_chunks, judge_model, client, eval_results_path, clean_rubrics):
    """Process a single chunk of examples for judging."""
    try:
        # Create thread-local client for this worker
        thread_client = client
        
        llm_judge = LLM_Judge(thread_client, judge_model)
        chunk_results = []
        
        print(f"Starting chunk {chunk_id}/{total_chunks} with {len(chunk_data)} examples")
        
        for i, example in enumerate(chunk_data):
            print(f"\rChunk {chunk_id}: Processing {i+1}/{len(chunk_data)} (Task ID: {example['task_id']})", end="")
            
            accuracy = 1
            total_weight = 0
            met_weight = 0
            example_results = []

            working_item = next((item for item in clean_rubrics if item['task_id'] == example['task_id']), None)
            if working_item is None:
                print(f"Warning: No item found for task ID {example['task_id']}")
                continue

            task_turns = example['turns']
            clean_rubrics_turns = working_item['turns']
        
            for task_turn, clean_rubrics_turn in zip(task_turns, clean_rubrics_turns):
                rubric_turn_results = []
                task_rubrics = clean_rubrics_turn['rubrics']
                for rubric_id, rubric in task_rubrics.items():
                    individual_rubric_results = {}
                    # Add retry logic for LLMasJudge
                    max_retries = 3
                    judge_result = None
                    for attempt in range(max_retries):
                        try:
                            judge_result = llm_judge.LLMasJudge(
                                task_turn['prompt'],
                                task_turn["golden_answer"], 
                                task_turn["model_answer"], 
                                rubric['description']
                            )
                            if judge_result:
                                break
                            else:
                                print(f"\nAttempt {attempt + 1}: LLMasJudge returned None, retrying...")
                                time.sleep(1)
                        except Exception as e:
                            print(f"\nAttempt {attempt + 1}: Error in LLMasJudge: {e}")
                            time.sleep(1)
                            if attempt == max_retries - 1:
                                print(f"\nFailed to get judge result after {max_retries} attempts")
                                judge_result = '{"explanation": "Failed to get judge result", "judge_result": "Not Met"}'
                    
                    individual_rubric_results['rubric_id'] = rubric_id
                    individual_rubric_results['judge_result'] = judge_result
                    rubric_turn_results.append(individual_rubric_results)
                    judge_result_parsed = json.loads(judge_result)

                    weight = rubric['weight']
                    total_weight += weight
                    if judge_result_parsed["judge_result"] == "Met":
                        met_weight += weight
                    if judge_result_parsed["judge_result"] == "Not Met" and weight >= 4:
                        accuracy = 0

                example_results.append(rubric_turn_results)
                
            score = met_weight/total_weight if total_weight > 0 else 0
            
            # print(f"Example results: {example_results}")
            # print(f"Score: {score}")
            # print(f"Accuracy: {accuracy}")
            # print(f"Total evaluated turns: {len(example_results)}")
            # exit()

            llm_judge_result = {
                'task_id': example['task_id'],
                'turncase': example['turncase'],
                'eval_focus': example['eval_focus'],
                'prompt_category': example['prompt_category'],
                'judge_result': example_results,
                'score': score,
                'accuracy': accuracy
            }
            
            chunk_results.append(llm_judge_result)
        
        print(f"\nChunk {chunk_id} completed: {len(chunk_results)} results processed")
        
        # Save chunk results to file immediately
        chunk_results_file = os.path.join(os.path.dirname(eval_results_path), f"chunk_{chunk_id}_judge_results.json")
        with open(chunk_results_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_results, f, ensure_ascii=False, indent=2)
        
        print(f"Chunk {chunk_id} results saved to: {chunk_results_file}")
        return chunk_results
        
    except Exception as e:
        print(f"\nFatal error in chunk {chunk_id}: {e}")
        return []

def combine_chunk_results(chunk_files, final_results_file):
    """Combine all chunk results into a single file."""
    print(f"\nCombining {len(chunk_files)} chunk files...")
    
    all_results = []
    for chunk_file in chunk_files:
        if os.path.exists(chunk_file):
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_results = json.load(f)
                    all_results.extend(chunk_results)
                    print(f"Added {len(chunk_results)} results from {chunk_file}")
            except Exception as e:
                print(f"Warning: Could not read chunk file {chunk_file}: {e}")
        else:
            print(f"Warning: Chunk file {chunk_file} does not exist")
    
    if not all_results:
        print("ERROR: No results found in any chunk files!")
        print("Available chunk files:")
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                print(f"  - {chunk_file} (exists)")
            else:
                print(f"  - {chunk_file} (missing)")
        return []
    
    # Sort by task_id to maintain original order
    all_results.sort(key=lambda x: x.get('task_id', ''))
    
    # Save combined results
    with open(final_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Combined {len(all_results)} total results into {final_results_file}")
    return all_results

def grade_responses(model_response_path, eval_results_path, judge_model, client, num_workers=1, clean_rubrics_path=None):
    """
    Grade model responses using LLM judge with parallel processing.
    
    Args:
        model_response_path: Path to model response file
        eval_results_path: Path to save evaluation results
        judge_model: Model to use for judging
        client: OpenAI client
        num_workers: Number of parallel workers for processing
    """
    print(f"Loading model responses from: {model_response_path}")
    with open(model_response_path, "r") as f:
        model_response = json.load(f)
    
    print(f"Using judge model: {judge_model}")
    print(f"Using {num_workers} worker(s) for parallel processing")
    
    with open(clean_rubrics_path, "r") as f:
        clean_rubrics = json.load(f)
    
    # Check for already processed examples
    processed_tasks = set()
    if eval_results_path and os.path.exists(eval_results_path):
        try:
            with open(eval_results_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                processed_tasks = {result['task_id'] for result in existing_results}
                print(f"Found {len(processed_tasks)} already processed tasks")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read existing results file: {e}")
            processed_tasks = set()
    
    # Filter out already processed examples
    remaining_examples = [ex for ex in model_response if ex['task_id'] not in processed_tasks]
    print(f"Processing {len(remaining_examples)} remaining examples")
    
    if not remaining_examples:
        print("All examples already processed!")
        return eval_results_path
    
    # Calculate chunk size and create chunks
    chunk_size = max(1, math.ceil(len(remaining_examples) / num_workers))
    chunks = []
    for i in range(0, len(remaining_examples), chunk_size):
        chunks.append(remaining_examples[i:i + chunk_size])
    
    print(f"Divided into {len(chunks)} chunks of ~{chunk_size} examples each")
    
    # Setup output directory
    output_dir = os.path.dirname(eval_results_path)
    os.makedirs(output_dir, exist_ok=True)
    
    if num_workers == 1:
        # Sequential processing for single worker
        print("Processing sequentially...")
        chunk_results_file = os.path.join(output_dir, f"chunk_1_judge_results.json")
        all_results = process_chunk_judging(
            remaining_examples, 1, 1, judge_model, client, eval_results_path, clean_rubrics
        )
        
        # Copy to final results file
        if all_results:
            shutil.copy2(chunk_results_file, eval_results_path)
            print(f"Results saved to: {eval_results_path}")
    else:
        # Parallel processing with multiple workers
        print(f"Processing {len(chunks)} chunks in parallel...")
        
        chunk_files = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit chunk processing tasks
            future_to_chunk = {}
            for i, chunk in enumerate(chunks):
                chunk_results_file = os.path.join(output_dir, f"chunk_{i+1}_judge_results.json")
                chunk_files.append(chunk_results_file)
                
                future = executor.submit(
                    process_chunk_judging, 
                    chunk, 
                    i+1, 
                    len(chunks), 
                    judge_model, 
                    client, 
                    eval_results_path,
                    clean_rubrics
                )
                future_to_chunk[future] = i+1
            
            # Wait for all chunks to complete
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    future.result()
                    print(f"Chunk {chunk_id} completed successfully")
                except Exception as e:
                    print(f"Chunk {chunk_id} failed: {e}")
        
        # Combine all chunk results
        all_results = combine_chunk_results(chunk_files, eval_results_path)
        
        # Clean up chunk files (optional)
        print("Cleaning up chunk files...")
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
                print(f"Removed: {chunk_file}")
    
    print(f"\nGrading completed!")
    print(f"Total results: {len(all_results) if 'all_results' in locals() else 'unknown'}")
    
    return eval_results_path

def analyze_results(eval_results_path):
    """
    Analyze evaluation results and print detailed statistics.
    
    Args:
        eval_results_path: Path to evaluation results file
    """
    print(f"\nAnalyzing results from: {eval_results_path}")
    
    if not os.path.exists(eval_results_path):
        print(f"Error: Results file {eval_results_path} does not exist!")
        return None
    
    with open(eval_results_path, "r") as f:
        eval_results = json.load(f)
    
    if not eval_results:
        print("Error: Results file is empty!")
        return None
    
    APR = 0
    ARS = 0
    domain_score = {}
    eval_focus_score = {}
    turncase_score = {}
    
    # STEM categories to combine
    stem_categories = ['chemistry', 'maths', 'biology', 'engineering', 'physics']
    
    for item in eval_results:
        APR += item['accuracy']
        ARS += item['score']
        
        # Domain scores based on prompt_category - combine STEM categories
        category = item['prompt_category']
        if category in stem_categories:
            category = 'STEM'
        
        if category not in domain_score:
            domain_score[category] = {'scores': [], 'accuracies': []}
        domain_score[category]['scores'].append(item['score'])
        domain_score[category]['accuracies'].append(item['accuracy'])
        
        # Eval focus scores
        if item['eval_focus'] not in eval_focus_score:
            eval_focus_score[item['eval_focus']] = {'scores': [], 'accuracies': []}
        eval_focus_score[item['eval_focus']]['scores'].append(item['score'])
        eval_focus_score[item['eval_focus']]['accuracies'].append(item['accuracy'])
        
        # Turncase scores
        if item['turncase'] not in turncase_score:
            turncase_score[item['turncase']] = {'scores': [], 'accuracies': []}
        turncase_score[item['turncase']]['scores'].append(item['score'])
        turncase_score[item['turncase']]['accuracies'].append(item['accuracy'])
    
    # Calculate overall metrics
    overall_apr = APR/len(eval_results)
    overall_ars = ARS/len(eval_results)
    
    # Print overall results
    print(f"\n=== OVERALL RESULTS ===")
    print(f"Overall Accuracy: {overall_apr:.4f}")
    print(f"Overall Score: {overall_ars:.4f}")
    print(f"Total Examples: {len(eval_results)}")
    
    # Print domain-wise results
    print(f"\n=== DOMAIN-WISE RESULTS ===")
    print("-" * 60)
    domainwise_score = {}
    for domain, data in sorted(domain_score.items()):
        avg_score = sum(data['scores']) / len(data['scores'])
        pass_rate = sum(data['accuracies']) / len(data['accuracies'])
        print(f"{domain:15}: Accuracy={pass_rate:.4f}, Score={avg_score:.4f} (n={len(data['scores'])})")
        domainwise_score[domain] = {'scores': avg_score, 'accuracies': pass_rate}
    
    # # Print turncase-wise results
    # print(f"\n=== TURNCASE-WISE RESULTS ===")
    # print("-" * 60)
    # turncasewise_score = {}
    # for turncase, data in sorted(turncase_score.items()):
    #     avg_score = sum(data['scores']) / len(data['scores'])
    #     pass_rate = sum(data['accuracies']) / len(data['accuracies'])
    #     print(f"{turncase:15}: Accuracy={pass_rate:.4f}, Score={avg_score:.4f} (n={len(data['scores'])})")
    #     turncasewise_score[turncase] = {'scores': data['scores'], 'accuracies': data['accuracies']}

    # Print eval focus results
    print(f"\n=== EVAL FOCUS RESULTS ===")
    print("-" * 60)
    eval_focuswise_score = {}
    for focus, data in sorted(eval_focus_score.items()):
        avg_score = sum(data['scores']) / len(data['scores'])
        pass_rate = sum(data['accuracies']) / len(data['accuracies'])
        print(f"{focus:15}: Accuracy={pass_rate:.4f}, Score={avg_score:.4f} (n={len(data['scores'])})")
        eval_focuswise_score[focus] = {'scores': avg_score, 'accuracies': pass_rate}
    # Return results for potential further use
    return {
        'overall_accuracy': overall_apr,
        'overall_score': overall_ars,
        'domain_results': domainwise_score,
        'eval_focus_results': eval_focuswise_score,
        'total_examples': len(eval_results)
    }

def main():
    parser = argparse.ArgumentParser(description='Grade model responses and analyze results')
    parser.add_argument('--model_response', '-m', required=True, 
                       help='Path to model response JSON file')
    parser.add_argument('--eval_results', '-e', required=True,
                       help='Path to save evaluation results JSON file')
    parser.add_argument('--eval_summary', '-s', required=True,
                       help='Path to save evaluation summary JSON file')
    parser.add_argument('--judge_model', '-j', default='openai/o4-mini',
                       help='Model to use for judging (default: openai/o4-mini)')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of parallel workers for processing (default: 1)')
    parser.add_argument('--skip_grading', action='store_true',
                       help='Skip grading and only analyze existing results')
    parser.add_argument('--skip_analysis', action='store_true',
                       help='Skip analysis and only perform grading')
    parser.add_argument('--clean_rubrics', '-c', required=True,
                       help='Path to clean rubrics JSON file')
    
    args = parser.parse_args()
    
    # Validate num_workers
    if args.num_workers < 1:
        print("Error: num_workers must be at least 1")
        return
    
    # Setup client
    client = setup_client()
    
    # Perform grading if not skipped
    if not args.skip_grading:
        print("=== STARTING GRADING PROCESS ===")
        grade_responses(args.model_response, args.eval_results, args.judge_model, client, args.num_workers, args.clean_rubrics)

    
    # Perform analysis if not skipped
    if not args.skip_analysis:
        print("\n=== STARTING ANALYSIS PROCESS ===")
        analysis_results = analyze_results(args.eval_results)
        with open(args.eval_summary, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    print("\n=== EVALUATION COMPLETE ===")

if __name__ == "__main__":
    main() 