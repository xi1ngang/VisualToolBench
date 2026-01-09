from PIL import Image
import json
import os
from tools import VisionTools
from typing import Union, List
import litellm
from utils import encode_image_to_base64
from functools import partial
from prompt import system_prompt_low, system_prompt_mid, system_prompt_high


def FunC_with_tools(question_id, prompt, image_list, max_tool_calls: int =20, model_name: str = "gemini/gemini-2.0-flash", tool_observation_save_path: str = None, system_prompt_level: str = "high"):
    tool_calls = True
    i = 0 # tool call counter 
    tool_use_list = []
    content_list = []
    
    python_img_tool = partial(VisionTools.python_image_processing, processed_image_save_path=tool_observation_save_path)
    
    available_functions = {
        "python_image_processing": python_img_tool,
        "python_interpreter": VisionTools.python_interpreter,
        "google_search": VisionTools.google_search,
        "browser_get_page_text": VisionTools.browser_get_page_text,
        "historical_weather": VisionTools.historical_weather,
        "calculator": VisionTools.safe_calculator,
    }  
    
    # system prompt three categories: make them stronger and stronger
    sys_map = {"low": system_prompt_low, "mid": system_prompt_mid, "high": system_prompt_high}
    messages = []
    if system_prompt_level in sys_map:
        messages.append({"role": "system", "content": sys_map[system_prompt_level]})

    # Handle both string prompts and conversation history
    if isinstance(prompt, str):
        # Single string prompt - original behavior
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
        
    elif isinstance(prompt, list):
        # Conversation history - append all messages and add current turn with images
        messages.extend(prompt)
        
        # Add current turn with images if any
        if image_list:
            # Check if the last message is from user and add images to it
            if messages and messages[-1]["role"] == "user":
                # If last message already has content, append images to it
                if isinstance(messages[-1]["content"], list):
                    # Multi-modal content, add images
                    for img_path in image_list:
                        try:
                            encoded_image, detected_format = encode_image_to_base64(img_path)
                            messages[-1]["content"].append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:{detected_format};base64,{encoded_image}"}
                            })
                        except Exception as e:
                            print(f"Error encoding image {img_path}: {e}")
                else:
                    # Text content, convert to multi-modal and add images
                    original_content = messages[-1]["content"]
                    messages[-1]["content"] = [{"type": "text", "text": original_content}]
                    for img_path in image_list:
                        try:
                            encoded_image, detected_format = encode_image_to_base64(img_path)
                            messages[-1]["content"].append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:{detected_format};base64,{encoded_image}"}
                            })
                        except Exception as e:
                            print(f"Error encoding image {img_path}: {e}")
            else:
                # No user message yet, create one with images
                new_user_content = []
                if image_list:
                    for img_path in image_list:
                        try:
                            encoded_image, detected_format = encode_image_to_base64(img_path)
                            new_user_content.append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:{detected_format};base64,{encoded_image}"}
                            })
                        except Exception as e:
                            print(f"Error encoding image {img_path}: {e}")
                    if new_user_content:
                        messages.append({"role": "user", "content": new_user_content})
    else:
        # Fallback: convert to string
        print(f"Warning: prompt is neither string nor list, converting to string: {type(prompt)}")
        user_content = [{"type": "text", "text": str(prompt)}]
        
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
    
    while tool_calls and i < max_tool_calls:
        print(f"\nRound {i}")
        # print(image_list)
        tools = VisionTools.get_tools(image_list, tool_observation_save_path)

        # Implement retry logic with maximum 3 trials
        max_trials = 1
        response = None
        
        for trial in range(max_trials):
            try:
                # Models that require temperature=1.0 (reasoning models)
                reasoning_models = ["o3", "o1", "o1-pro", "o4-mini", "gpt-5", "gpt-5-mini"]
                is_reasoning_model = any(rm in model_name for rm in reasoning_models)
                
                if "thinking" in model_name:
                    # Handle thinking models with extended budget
                    base_model_name = model_name.replace("-thinking", "")
                    print(f"Using thinking model: {base_model_name} (trial {trial + 1}/{max_trials})")
                    response = litellm.completion(
                        model=base_model_name,
                        messages=messages,
                        thinking={"type": "enabled", "budget_tokens": 5000},
                        tools=tools,
                        temperature=1.0,
                        tool_choice="auto",
                    )
                elif "gpt-5-think" in model_name:
                    print(f"Using model: {model_name} (trial {trial + 1}/{max_trials})")
                    response = litellm.completion(
                        model="gpt-5",
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        reasoning_effort="high",
                    )
                else:
                    print(f"Using model: {model_name} (trial {trial + 1}/{max_trials})")
                    response = litellm.completion(
                        model=model_name,
                        messages=messages,
                        tools=tools,
                        temperature=1.0 if is_reasoning_model else 0.0,
                        tool_choice="auto",
                    )
                
                # If we get here, the API call was successful
                print(f"Model inference successful on trial {trial + 1}")
                break
                
            except Exception as e:
                print(f"Trial {trial + 1} failed: {e}")
                if trial == max_trials - 1:
                    # All trials failed
                    error_msg = f"Model inference failed after {max_trials} trials. Last error: {e}"
                    print(error_msg)
                    return error_msg, i, tool_use_list, content_list
                else:
                    # Wait a bit before retrying
                    import time
                    time.sleep(2)
        
        if response is None:
            error_msg = f"Model inference failed after {max_trials} trials."
            print(error_msg)
            return error_msg, i, tool_use_list, content_list
        
        response_message = response.choices[0].message
        # print(f"response_message: {response_message}")

        tool_calls = response_message.tool_calls

        response_content = response_message.content
        content_list.append({"step": i, "content": response_content})

        if not tool_calls:
            return response_message.content, i, tool_use_list, content_list

        messages.append(response_message)


        # Step 4: send the info for each function call and function response to the model
        for tc in tool_calls:
            function_name = tc.function.name
            print(f"calling tool {function_name}")
            # try:
            function_to_call = available_functions[function_name]
            function_args = json.loads(tc.function.arguments)
            print(f"function_args: {function_args}")
            try:
                observation = function_to_call(**function_args)
            except Exception as e:
                print(f"Error calling tool {function_name}: {e}")
                observation = {"error": str(e)}
            # print(f"observation: {observation}")
            print(f"observation: {observation}")
            tool_use_info = {}
            tool_use_info['function_name'] = function_name
            # tool_use_info['reasoning'] = function_args['reasoning']
            tool_use_info['function_args'] = function_args
            tool_use_info['observation'] = observation
    
            tool_use_list.append(tool_use_info)

            # Handle transformed images from python_image_processing
            if function_name == "python_image_processing" and observation.get('output_paths', []):
                transformed_image_paths = observation['output_paths']
                print(f"transformed_image_paths: {transformed_image_paths}")
                
                # Add all transformed images to the image list
                for transformed_image_path in transformed_image_paths:
                    image_list.append(transformed_image_path)
                
                # Create content with all transformed images
                transformed_content = [
                    {"type": "text", "text": f"Here are the transformed images from the tool call {tc.id}."}
                ]
                
                # Add each transformed image to the content
                for transformed_image_path in transformed_image_paths:
                    try:
                        encoded_observation, detected_format = encode_image_to_base64(transformed_image_path)
                        transformed_image_content = {
                            "type": "image_url", 
                            "image_url": {"url": f"data:{detected_format};base64,{encoded_observation}"}
                        }
                        transformed_content.append(transformed_image_content)
                    except Exception as e:
                        print(f"Error processing image {transformed_image_path}: {e}")
                
                messages.append(
                {
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(observation),
                }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": transformed_content,
                    }
                )
            else:
                messages.append(
                {
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(observation),
                }
            )
        # Increment round counter after processing all tool calls
        i += 1
            
            
    return "Model hit the maximum number of tool calls", i, tool_use_list, content_list