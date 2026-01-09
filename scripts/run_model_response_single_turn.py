"""
Model response generation script for vision tool use evaluation.
Uses LiteLLM for unified access to multiple model providers.

Set API keys in your .env file or environment:
- OPENAI_API_KEY for OpenAI models (gpt-4o, o3, etc.)
- GEMINI_API_KEY for Google models (gemini/gemini-2.0-flash, etc.)
- ANTHROPIC_API_KEY for Anthropic models (claude-sonnet-4-20250514, etc.)
"""
import os
import argparse
from dotenv import load_dotenv
import json
from datetime import datetime
import litellm
from model_inference import FunC_with_tools
from utils import resize_image_for_llama
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import math

# Load environment variables
load_dotenv()

# LiteLLM will automatically use these environment variables:
# - OPENAI_API_KEY for OpenAI models (gpt-4o, o3, etc.)
# - GEMINI_API_KEY or GOOGLE_API_KEY for Gemini models (gemini/gemini-2.0-flash, etc.)
# - ANTHROPIC_API_KEY for Anthropic models (claude-sonnet-4-20250514, etc.)

def process_chunk(chunk_data, chunk_id, total_chunks, model_name, tool_use, max_tool_calls, 
                  system_prompt_level, save_path, chunk_results_file):
    """Process a single chunk of examples."""
    try:
        chunk_results = []
        
        print(f"Starting chunk {chunk_id}/{total_chunks} with {len(chunk_data)} examples")
        
        for i, example in enumerate(chunk_data):
            try:
                # Setup image path - handle both list and string formats
                current_dir = os.path.dirname(os.path.abspath(__file__))
                image_paths_raw = example.get("image_paths", example.get("image", []))
                
                # Handle list of paths or single path
                if isinstance(image_paths_raw, list):
                    image_paths_raw = image_paths_raw[0] if image_paths_raw else None
                
                # Handle absolute vs relative paths
                if image_paths_raw:
                    if os.path.isabs(image_paths_raw):
                        image_path = image_paths_raw
                    else:
                        # Resolve relative to project root (parent of scripts/)
                        project_root = os.path.dirname(current_dir)
                        image_path = os.path.join(project_root, image_paths_raw)
                else:
                    image_path = None
                    
                if image_path and "llama" in model_name:
                    image_path = resize_image_for_llama(image_path)

                # Setup save paths
                tool_use_observation_save_path = os.path.join(save_path, f"chunk_{chunk_id}_task_{example['id']}")
                os.makedirs(save_path, exist_ok=True)

                # Handle prompt - can be string or list (turn_prompts)
                prompt = example.get("turn_prompts", example.get("prompt", ""))
                if isinstance(prompt, list):
                    prompt = prompt[0] if prompt else ""
                    
                image_list = [image_path] if image_path else []
                
                if tool_use:
                    model_answer, num_tool_calls, tool_use_list_observation, content_list = FunC_with_tools(
                        question_id=example["id"],
                        prompt=prompt,
                        image_list=image_list,
                        max_tool_calls=max_tool_calls,
                        model_name=model_name,
                        tool_observation_save_path=tool_use_observation_save_path,
                        system_prompt_level=system_prompt_level
                    )
                else:
                    try:
                        from utils import encode_image_to_base64
                        messages = []
                        user_content = [{"type": "text", "text": prompt}]
                        if image_list:
                            # Add each image to the content
                            for img_path in image_list:
                                try:
                                    encoded_image, detected_format = encode_image_to_base64(img_path)
                                    user_content.append({
                                        "type": "image_url", 
                                        "image_url": {"url": f"data:{detected_format};base64,{encoded_image}"}
                                    })
                                except Exception as e:
                                    print(f"Error encoding image {img_path}: {e}")
                        messages.append({"role": "user", "content": user_content})
                        
                        # Models that require temperature=1.0 (reasoning models)
                        reasoning_models = ["o3", "o1", "o1-pro", "o4-mini", "gpt-5", "gpt-5-mini"]
                        is_reasoning_model = any(rm in model_name for rm in reasoning_models)
                        
                        response = litellm.completion(
                            model=model_name,
                            messages=messages,
                            temperature=1.0 if is_reasoning_model else 0.0,
                        )   
                        model_answer = response.choices[0].message.content
                        num_tool_calls = 0
                        tool_use_list_observation = {}
                        content_list = {}
                    except Exception as e:
                        print(f"Error in API call for task {example['id']}: {e}")
                        model_answer = f"API Error: {str(e)}"
                        num_tool_calls = 0
                        tool_use_list_observation = {}
                        content_list = {}

                # Handle both old and new dataset formats
                task_id = example["id"]
                eval_result = {
                    'id': task_id,
                    'task_id': task_id,  # Alias for compatibility with judge scripts
                    'prompt': prompt,
                    'image': image_paths_raw,
                    'turncase': example.get("turncase", "single-turn"),
                    'eval_focus': example.get("eval_focus", ""),
                    'prompt_category': example.get("prompt_category", ""),
                    'golden_answer': example.get("turn_golden_answers", [example.get("golden_answer", "")])[0] if isinstance(example.get("turn_golden_answers"), list) else example.get("golden_answer", ""),
                    'tool_trajectory': example.get("turn_tool_trajectories", [example.get("tool_trajectory", "")])[0] if isinstance(example.get("turn_tool_trajectories"), list) else example.get("tool_trajectory", ""),
                    'rubrics': example.get("rubrics_by_turn", [example.get("rubrics", "")])[0] if isinstance(example.get("rubrics_by_turn"), list) else example.get("rubrics", ""),
                    'model_answer': model_answer,
                    'num_tool_calls': num_tool_calls,
                    'tool_use_list': tool_use_list_observation,
                    'content_list': content_list
                }
                
                chunk_results.append(eval_result)
                print(f"\rChunk {chunk_id}: Processed {i+1}/{len(chunk_data)} (Task ID: {example['id']})", end="")
                
            except Exception as e:
                error_id = example.get('id', f'error_{i}')
                print(f"\nError processing task {error_id} in chunk {chunk_id}: {e}")
                # Add error result to maintain task count
                error_result = {
                    'id': error_id,
                    'task_id': error_id,  # Alias for compatibility with judge scripts
                    'prompt': example.get("prompt", ""),
                    'image': example.get("image", ""),
                    'turncase': example.get("turncase", ""),
                    'eval_focus': example.get("eval_focus", ""),
                    'prompt_category': example.get("prompt_category", ""),
                    'golden_answer': example.get("golden_answer", ""),
                    'tool_trajectory': example.get("tool_trajectory", ""),
                    'rubrics': example.get("rubrics", ""),
                    'model_answer': f"Processing Error: {str(e)}",
                    'num_tool_calls': 0,
                    'tool_use_list': {},
                    'content_list': {}
                }
                chunk_results.append(error_result)
        
        # Save chunk results to separate file
        with open(chunk_results_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_results, f, ensure_ascii=False, indent=2)
        
        print(f"\nChunk {chunk_id} completed: {len(chunk_results)} results saved to {chunk_results_file}")
        return chunk_results
        
    except Exception as e:
        print(f"\nFatal error in chunk {chunk_id}: {e}")
        # Save partial results if possible
        if chunk_results:
            with open(chunk_results_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_results, f, ensure_ascii=False, indent=2)
        return []

def combine_chunk_results(chunk_files, final_results_file):
    """Combine all chunk results into a single file and merge with existing results."""
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
    
    if not all_results:
        print("ERROR: No results found in any chunk files!")
        return []
    
    # Sort by id to maintain original order
    all_results.sort(key=lambda x: x.get('id', ''))
    
    # Merge with existing results if file exists
    final_results = all_results
    if os.path.exists(final_results_file):
        try:
            with open(final_results_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            # Create a set of existing task IDs for quick lookup
            existing_ids = {result['id'] for result in existing_results}
            
            # Add new results, avoiding duplicates
            for result in all_results:
                if result['id'] not in existing_ids:
                    existing_results.append(result)
                    existing_ids.add(result['id'])
            
            final_results = existing_results
            print(f"Merged with {len(existing_results)} existing results")
            
        except Exception as e:
            print(f"Warning: Could not read existing results file: {e}")
    
    # Save combined results
    with open(final_results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"Combined {len(final_results)} total results into {final_results_file}")
    return final_results

def cleanup_orphaned_task_folders(save_path: str, final_results_file: str) -> None:
    """
    Clean up task folders that don't correspond to tasks in the final results.
    This handles cases where the script was interrupted and some task folders were created
    but their results weren't saved to the final results file.
    
    Args:
        save_path: Directory containing task folders
        final_results_file: Path to the final results file
    """
    if not os.path.exists(save_path) or not os.path.exists(final_results_file):
        return
    
    try:
        # Load final results to get valid task IDs
        with open(final_results_file, 'r', encoding='utf-8') as f:
            final_results = json.load(f)
        
        valid_ids = {str(result['id']) for result in final_results}
        print(f"Found {len(valid_ids)} valid task IDs in final results")
        
        # Find all task folders with the pattern: chunk_X_task_TASKID
        task_folders = []
        for item in os.listdir(save_path):
            item_path = os.path.join(save_path, item)
            if os.path.isdir(item_path) and item.startswith('chunk_') and '_task_' in item:
                task_folders.append(item)
        
        print(f"Found {len(task_folders)} task folders")
        
        # Remove orphaned task folders
        removed_count = 0
        for folder in task_folders:
            # Extract task ID from folder name (e.g., "chunk_1_task_68658a711603983919432616" -> "68658a711603983919432616")
            if '_task_' in folder:
                id = folder.split('_task_')[1]  # Get everything after "_task_"
                
                if id not in valid_ids:
                    folder_path = os.path.join(save_path, folder)
                    try:
                        shutil.rmtree(folder_path)
                        print(f"Removed orphaned task folder: {folder}")
                        removed_count += 1
                    except Exception as e:
                        print(f"Warning: Could not remove folder {folder}: {e}")
        
        if removed_count > 0:
            print(f"Cleaned up {removed_count} orphaned task folders")
        else:
            print("No orphaned task folders found")
            
    except Exception as e:
        print(f"Warning: Could not clean up orphaned task folders: {e}")

def get_model_response(
    dataset_path: str,
    model_name: str,
    tool_use: bool = True,
    save_path: str = None,
    results_file: str = None,
    system_prompt_level: str = "high",
    max_tool_calls: int = 20,
    num_workers: int = 1
):
    """Run evaluation on specified dataset and model with chunk-based parallel processing."""
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Starting evaluation with model: {model_name}")
    print(f"Using {num_workers} worker(s) for parallel processing")

    total = len(dataset)
    print(f"Total examples: {total}")

    # IMPROVED: First consolidate all existing chunk results to get complete picture
    processed_tasks = set()
    existing_results = []
    
    # Check final results file first
    if results_file and os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                processed_tasks = {result['id'] for result in existing_results}
                print(f"Found {len(processed_tasks)} already processed tasks in final results file")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read final results file: {e}")
    
    # Check for existing chunk files and consolidate them
    if os.path.exists(save_path):
        chunk_files = []
        for file in os.listdir(save_path):
            if file.startswith('chunk_') and file.endswith('_results.json'):
                chunk_files.append(os.path.join(save_path, file))
        
        if chunk_files:
            print(f"Found {len(chunk_files)} existing chunk files, consolidating...")
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_results = json.load(f)
                        # Add new results from chunks
                        for result in chunk_results:
                            if result['id'] not in processed_tasks:
                                existing_results.append(result)
                                processed_tasks.add(result['id'])
                        print(f"Added {len(chunk_results)} results from {chunk_file}")
                except Exception as e:
                    print(f"Warning: Could not read chunk file {chunk_file}: {e}")
            
            # Save consolidated results to final file
            if existing_results:
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)
                print(f"Consolidated {len(existing_results)} total results to {results_file}")
    
    print(f"Total already processed tasks: {len(processed_tasks)}")

    # Filter out already processed examples
    remaining_examples = [ex for ex in dataset if ex['id'] not in processed_tasks]
    print(f"Processing {len(remaining_examples)} remaining examples")

    if not remaining_examples:
        print("All examples already processed!")
        # Clean up any orphaned task folders before exiting
        cleanup_orphaned_task_folders(save_path, results_file)
        return

    # ADD CLEANUP HERE - before starting new work
    print("Cleaning up orphaned task folders from previous runs...")
    cleanup_orphaned_task_folders(save_path, results_file)

    # Calculate chunk size and create chunks
    chunk_size = max(1, math.ceil(len(remaining_examples) / num_workers))
    chunks = []
    for i in range(0, len(remaining_examples), chunk_size):
        chunks.append(remaining_examples[i:i + chunk_size])
    
    print(f"Divided into {len(chunks)} chunks of ~{chunk_size} examples each")

    if num_workers == 1:
        # Sequential processing for single worker
        print("Processing sequentially...")
        chunk_results_file = os.path.join(save_path, f"chunk_1_results.json")
        all_results = process_chunk(
            remaining_examples, 1, 1, model_name, tool_use, 
            max_tool_calls, system_prompt_level, save_path, chunk_results_file
        )
        
        # Merge with existing results and save to final file
        if all_results:
            # Add new results to existing ones
            for result in all_results:
                if result['id'] not in processed_tasks:
                    existing_results.append(result)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=2)
            print(f"Results merged and saved to: {results_file}")
            
            # Clean up orphaned task folders after saving results
            cleanup_orphaned_task_folders(save_path, results_file)
    else:
        # Parallel processing with multiple workers
        print(f"Processing {len(chunks)} chunks in parallel...")
        
        chunk_files = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit chunk processing tasks
            future_to_chunk = {}
            for i, chunk in enumerate(chunks):
                chunk_results_file = os.path.join(save_path, f"chunk_{i+1}_results.json")
                chunk_files.append(chunk_results_file)
                
                future = executor.submit(
                    process_chunk, 
                    chunk, 
                    i+1, 
                    len(chunks), 
                    model_name, 
                    tool_use, 
                    max_tool_calls, 
                    system_prompt_level, 
                    save_path, 
                    chunk_results_file
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
        
        # Combine all chunk results and merge with existing results
        all_results = combine_chunk_results(chunk_files, results_file)
        
        # Clean up chunk files (optional)
        print("Cleaning up chunk files...")
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
                print(f"Removed: {chunk_file}")
        
        # Clean up orphaned task folders after saving results
        cleanup_orphaned_task_folders(save_path, results_file)

    print(f"\nEvaluation complete for model: {model_name}")
    print(f"Total results: {len(existing_results) + len(all_results) if 'all_results' in locals() else len(existing_results)}")

def main():
    parser = argparse.ArgumentParser(description='Run model evaluation on vision tool use benchmark')
    parser.add_argument('--model', '-m', required=True, 
                       help='Model name to evaluate (e.g., openai/gpt-4o-mini)')
    parser.add_argument('--dataset', '-d', required=True,
                       help='Path to dataset JSON file')
    parser.add_argument('--output_dir', '-o', default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--tool_use', action='store_true', default=False,
                       help='Enable tool use (default: False)')
    parser.add_argument('--no_tool_use', dest='tool_use', action='store_false',
                       help='Disable tool use')
    parser.add_argument('--system_prompt_level', default='high',
                       choices=['low', 'medium', 'high'],
                       help='System prompt level (default: high)')
    parser.add_argument('--max_tool_calls', type=int, default=20,
                       help='Maximum number of tool calls (default: 20)')
    parser.add_argument('--trial', type=int, default=1,
                       help='Trial number (default: 1)')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of parallel workers for processing (default: 1)')
    
    args = parser.parse_args()
    
    # Validate num_workers
    if args.num_workers < 1:
        print("Error: num_workers must be at least 1")
        return
    
    # Setup output paths
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create descriptive filename
    tool_status = "w_tool" if args.tool_use else "wo_tool"
    
    save_path = os.path.join(
        args.output_dir, 
        f"{args.model}_{tool_status}_system_{args.system_prompt_level}_max_tool_calls_{args.max_tool_calls}_trial_{args.trial}"
    )
    
    results_file = f"{save_path}.json"
    
    print(f"=== EVALUATION SETUP ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Tool use: {args.tool_use}")
    print(f"System prompt level: {args.system_prompt_level}")
    print(f"Max tool calls: {args.max_tool_calls}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Trial: {args.trial}")
    print(f"Output: {results_file}")
    print(f"Save path: {save_path}")
    print("=" * 50)
    
    # Run evaluation
    start_time = time.time()
    try:
        get_model_response(
            dataset_path=args.dataset,
            model_name=args.model,
            tool_use=args.tool_use,
            save_path=save_path,
            results_file=results_file,
            system_prompt_level=args.system_prompt_level,
            max_tool_calls=args.max_tool_calls,
            num_workers=args.num_workers
        )
        end_time = time.time()
        print(f"\n=== EVALUATION COMPLETE ===")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n=== EVALUATION FAILED ===")
        print(f"Error: {e}")
        # Save error info
        error_file = f"{save_path}_error.txt"
        with open(error_file, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Time: {datetime.now()}\n")
        print(f"Error details saved to: {error_file}")

if __name__ == "__main__":
    main()