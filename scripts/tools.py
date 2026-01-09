import re
import os
import sys
import io
import json
import math
import ast
import operator as op
import contextlib
import traceback
from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict
from datetime import datetime, timedelta

# Image processing imports
from PIL import Image, ImageOps, ImageEnhance

# Data processing imports
import numpy as np
import pandas as pd
import requests
from scipy import stats
import simpy
from tabulate import tabulate
from bs4 import BeautifulSoup
import yfinance as yf
import cv2

# Local imports
from utils import list_transformed_pngs
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Mathematical operators for safe calculator
OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg
}

# Supported math functions
MATH_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp
}

# Regex for finding transformed image files
_IMG_RE = re.compile(r"^transformed_image_(\d+)\.png$", re.IGNORECASE)


class VisionTools:
    """Collection of vision and utility tools."""
    
    @staticmethod
    def python_image_processing(code: str, processed_image_save_path: Union[str, Path, None] = None) -> Dict[str, str]:
        """
        Execute LLM-supplied Python code for image processing.
        
        Parameters
        ----------
        code : str
            Python code to execute
        processed_image_save_path : str | Path | None
            Directory to save processed images
            
        Returns
        -------
        dict
            Contains execution status, output, and file paths
        """
        # Setup save directory
        save_dir = Path(processed_image_save_path).expanduser() if processed_image_save_path else Path.cwd()
        save_dir.mkdir(parents=True, exist_ok=True)

        # Set matplotlib to non-interactive backend to prevent image display
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create a custom Image class that overrides show() method
        class SafeImage:
            def __init__(self, *args, **kwargs):
                self._image = Image.Image(*args, **kwargs)
            
            def __getattr__(self, name):
                # Delegate all other methods to the original Image
                return getattr(self._image, name)
            
            def show(self, *args, **kwargs):
                # No-op function to prevent image display
                pass
            
            @staticmethod
            def open(*args, **kwargs):
                # Create a new SafeImage instance
                img = Image.Image.open(*args, **kwargs)
                safe_img = SafeImage()
                safe_img._image = img
                return safe_img
        
        # Create safe matplotlib.pyplot with no-op show function
        import matplotlib.pyplot as plt
        class SafePyplot:
            def __getattr__(self, name):
                return getattr(plt, name)
            
            def show(self, *args, **kwargs):
                # No-op function to prevent plot display
                pass
        
        safe_plt = SafePyplot()

        # Build restricted execution environment
        safe_globals = {
            "__builtins__": {
                name: __builtins__[name]
                for name in (
                    "print", "range", "len", "min", "max", "abs",
                    "float", "int", "str", "list", "dict", "set",
                    "tuple", "enumerate", "zip", "map", "filter", "__import__"
                )
            },
            "Image": SafeImage,
            "np": np,
            "cv2": cv2,
            "plt": safe_plt,  # Use safe matplotlib.pyplot
            "PROCESSED_IMAGE_SAVE_PATH": save_dir,
        }
        safe_locals = {}

        # Capture stdout/stderr
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        ok = True

        # Track files before execution
        before_files = set(list_transformed_pngs(save_dir))

        # Execute code
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            try:
                exec(compile(code, "<llm_code>", "exec"), safe_globals, safe_locals)
            except Exception:
                ok = False
                traceback.print_exc(file=sys.stderr)

        # Find new files after execution
        after_files = set(list_transformed_pngs(save_dir))
        new_files = after_files - before_files

        # Get output paths
        output_paths = []
        for file_path in new_files:
            try:
                # Return the full path instead of relative path
                output_paths.append(str(file_path))
            except ValueError:
                output_paths.append(str(file_path))

        return {
            "ok": "true" if ok else "false",
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "output_path": output_paths[0] if output_paths else "",
            "output_paths": output_paths  # Return all paths
        }

    @staticmethod
    def python_interpreter(code: str) -> Dict[str, str]:
        """
        Run arbitrary Python code in a restricted environment.
        
        Parameters
        ----------
        code : str
            Python code to execute
            
        Returns
        -------
        dict
            Contains stdout and stderr output
        """
        # Set matplotlib to non-interactive backend to prevent image display
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create a custom Image class that overrides show() method
        class SafeImage:
            def __init__(self, *args, **kwargs):
                self._image = Image.Image(*args, **kwargs)
            
            def __getattr__(self, name):
                # Delegate all other methods to the original Image
                return getattr(self._image, name)
            
            def show(self, *args, **kwargs):
                # No-op function to prevent image display
                pass
            
            @staticmethod
            def open(*args, **kwargs):
                # Create a new SafeImage instance
                img = Image.Image.open(*args, **kwargs)
                safe_img = SafeImage()
                safe_img._image = img
                return safe_img
        
        # Create safe matplotlib.pyplot with no-op show function
        import matplotlib.pyplot as plt
        class SafePyplot:
            def __getattr__(self, name):
                return getattr(plt, name)
            
            def show(self, *args, **kwargs):
                # No-op function to prevent plot display
                pass
        
        safe_plt = SafePyplot()

        # Build restricted execution environment
        safe_globals = {
            "__builtins__": {
                "print": print,
                "range": range,
                "len": len,
                "min": min,
                "max": max,
                "abs": abs,
                "float": float,
                "int": int,
                "str": str,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "__import__": __import__,
            },
            "np": np,
            "pandas": pd,
            "pd": pd,
            "requests": requests,
            "scipy": stats,
            "sklearn": __import__("sklearn"),
            "simpy": simpy,
            "tabulate": tabulate,
            "BeautifulSoup": BeautifulSoup,
            "Image": SafeImage,  # Use safe Image class
            "plt": safe_plt,  # Use safe matplotlib.pyplot
        }
        safe_locals = {}

        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            try:
                exec(compile(code, "<llm_code>", "exec"), safe_globals, safe_locals)
            except Exception:
                traceback.print_exc()

        return {
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        }

    @staticmethod
    def _eval_ast(node):
        """Recursively evaluate an AST node for mathematical expressions."""
        if isinstance(node, ast.Constant):  # Python 3.8+ uses ast.Constant
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7 and earlier
            return node.n
        elif isinstance(node, ast.BinOp):
            return OPERATORS[type(node.op)](VisionTools._eval_ast(node.left),
                                          VisionTools._eval_ast(node.right))
        elif isinstance(node, ast.UnaryOp):
            return OPERATORS[type(node.op)](VisionTools._eval_ast(node.operand))
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name not in MATH_FUNCS:
                raise ValueError(f"Function {func_name} not allowed.")
            if len(node.args) != 1:
                raise ValueError("Only one argument permitted.")
            return MATH_FUNCS[func_name](VisionTools._eval_ast(node.args[0]))
        else:
            raise ValueError("Unsupported expression element.")

    @staticmethod
    def safe_calculator(expression: str) -> Dict[str, str]:
        """
        Safely evaluate mathematical expressions.
        
        Parameters
        ----------
        expression : str
            Mathematical expression to evaluate
            
        Returns
        -------
        dict
            Contains result or error message
        """
        try:
            tree = ast.parse(expression, mode="eval")
            result = VisionTools._eval_ast(tree.body)
            return {
                "result": str(result),
                "expression": expression,
                "status": "success"
            }
        except Exception as e:
            return {
                "result": "Error",
                "expression": expression,
                "status": "error",
                "error": str(e)
            }

    @staticmethod
    def google_search(query: str, num_results: int = 5) -> Dict[str, str]:
        """
        Perform Google search using SerpAPI.
        
        Parameters
        ----------
        query : str
            Search query
        num_results : int
            Number of results to return (max 10)
            
        Returns
        -------
        dict
            Search results or error information
        """
        try:
            api_key = os.getenv("SERP_API_KEY")
            
            if not api_key:
                return {
                    "results": [],
                    "query": query,
                    "status": "error",
                    "error": "SerpAPI key not configured"
                }
            
            url = "https://serpapi.com/search"
            params = {
                'api_key': api_key,
                'q': query,
                'num': min(num_results, 10),
                'engine': 'google'
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if 'organic_results' in data:
                for item in data['organic_results']:
                    results.append({
                        "title": item.get('title', ''),
                        "snippet": item.get('snippet', ''),
                        "url": item.get('link', ''),
                        "display_link": item.get('displayed_link', '')
                    })
            
            return {
                "results": results,
                "query": query,
                "status": "success",
                "total_results": data.get('search_information', {}).get('total_results', 0)
            }
        except Exception as e:
            return {
                "results": [],
                "query": query,
                "status": "error",
                "error": str(e)
            }

    @staticmethod
    def browser_get_page_text(url: str) -> Dict[str, str]:
        """
        Fetch and extract text content from a web page.
        
        Parameters
        ----------
        url : str
            URL to fetch
            
        Returns
        -------
        dict
            Page content or error information
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove non-content elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string
            elif soup.find('h1'):
                title = soup.find('h1').get_text()
            
            return {
                "url": url,
                "title": title,
                "text": text[:8000],
                "status": "success",
                "content_length": len(text)
            }
        except Exception as e:
            return {
                "url": url,
                "title": "",
                "text": "",
                "status": "error",
                "error": str(e)
            }

    @staticmethod
    def historical_weather(location: str, date: str) -> Dict[str, str]:
        """
        Get weather data for a location using OpenWeatherMap API.
        
        Parameters
        ----------
        location : str
            City name or coordinates
        date : str
            Date in YYYY-MM-DD format
            
        Returns
        -------
        dict
            Weather data or error information
        """
        try:
            api_key = os.getenv("OPENWEATHER_API_KEY")
            
            if not api_key:
                return {
                    "location": location,
                    "date": date,
                    "status": "error",
                    "error": "OpenWeatherMap API key not configured"
                }
            
            # Get coordinates for location
            geocode_url = "http://api.openweathermap.org/geo/1.0/direct"
            geocode_params = {
                'q': location,
                'limit': 1,
                'appid': api_key
            }
            
            geocode_response = requests.get(geocode_url, params=geocode_params, timeout=10)
            geocode_response.raise_for_status()
            geocode_data = geocode_response.json()
            
            if not geocode_data:
                return {
                    "location": location,
                    "date": date,
                    "status": "error",
                    "error": "Location not found"
                }
            
            lat = geocode_data[0]['lat']
            lon = geocode_data[0]['lon']
            
            # Get current weather (historical requires paid plan)
            weather_url = "https://api.openweathermap.org/data/2.5/weather"
            weather_params = {
                'lat': lat,
                'lon': lon,
                'appid': api_key,
                'units': 'metric'
            }
            
            weather_response = requests.get(weather_url, params=weather_params, timeout=10)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            return {
                "location": location,
                "date": date,
                "coordinates": f"{lat}, {lon}",
                "temperature": {
                    "current": f"{weather_data['main']['temp']:.1f}°C",
                    "feels_like": f"{weather_data['main']['feels_like']:.1f}°C",
                    "min": f"{weather_data['main']['temp_min']:.1f}°C",
                    "max": f"{weather_data['main']['temp_max']:.1f}°C"
                },
                "conditions": weather_data['weather'][0]['description'],
                "humidity": f"{weather_data['main']['humidity']}%",
                "wind_speed": f"{weather_data['wind']['speed']} m/s",
                "pressure": f"{weather_data['main']['pressure']} hPa",
                "status": "success",
                "note": "Current weather data (historical requires paid plan)"
            }
        except Exception as e:
            return {
                "location": location,
                "date": date,
                "status": "error",
                "error": str(e)
            }


    def get_tools(image_list, processed_image_save_path):
        """
        Build tool descriptor list for models.
        
        Parameters
        ----------
        image_list : List[str]
            List of available image files
        processed_image_save_path : str
            Directory for saving processed images
            
        Returns
        -------
        List[Dict]
            List of tool descriptors
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "python_image_processing",
                    "description": (
                        "Generate arbitrary Python code for image manipulation and save the transformed image as PNG.\n"
                        f"• Read one source image (your choice) from the working-directory file list: {image_list}.\n"
                        f"• Perform any image processing with PIL, NumPy, or OpenCV. You cannot use matplotlib to show the image.\n"
                        f"• You **must save** the transformed image as PNG to {processed_image_save_path} using the filename pattern "
                        "\"transformed_image_i.png\", where the counter **i starts at 0 and increments on each invocation** "
                        "so files are never overwritten. Example:\n"
                        f"    img.save(f\"{processed_image_save_path}/transformed_image_{{i}}.png\", \"PNG\")\n"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to run.",
                                "minLength": 1,
                                "maxLength": 5000
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "python_interpreter",
                    "description": (
                        "General-purpose Python interpreter. Run arbitrary Python code and capture stdout via print(). "
                        "Any exceptions are returned in stderr.\n\n"
                        "Pre-installed packages:\n"
                        "  • numpy\n"
                        "  • pandas\n"
                        "  • requests\n"
                        "  • scipy\n"
                        "  • scikit-learn\n"
                        "  • simpy\n"
                        "  • tabulate\n"
                        "  • beautifulsoup4\n"
                        "  • yfinance"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to run.",
                                "minLength": 1,
                                "maxLength": 5000
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "google_search",
                    "description": (
                        "Perform a Google search and return relevant results. "
                        "Useful for finding current information, news, or facts about topics."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to look up"
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results to return (1-10)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_get_page_text",
                    "description": (
                        "Fetch a web page and extract its text content. "
                        "Useful for reading articles, documentation, or any web page content."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL of the web page to fetch"
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "historical_weather",
                    "description": (
                        "Get historical weather data for a specific location and date. "
                        "Useful for analyzing past weather patterns or events."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name or coordinates (e.g., 'New York, NY' or '40.7128,-74.0060')"
                            },
                            "date": {
                                "type": "string",
                                "description": "Date in YYYY-MM-DD format"
                            }
                        },
                        "required": ["location", "date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "A calculator tool that can perform basic arithmetic operations including +, -, *, /, %, ^, sqrt, sin, cos, tan, log, exp, and parentheses.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "The expression to evaluate, e.g. \"2 * 3.14 * 5\"."}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

        return tools