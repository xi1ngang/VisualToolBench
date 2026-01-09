from PIL import Image
import io
import base64
from io import BytesIO
from pathlib import Path
from typing import List
import re
import mimetypes
import os
import tempfile

_IMG_RE = re.compile(r"^transformed_image_(\d+)\.png$", re.IGNORECASE)

def clean_unicode_escapes(text: str) -> str:
    """Convert Unicode escape sequences to readable characters."""
    return text.encode('utf-8').decode('unicode_escape')

# Apply to model responses
def clean_model_response(response: str) -> str:
    """Clean model response by handling Unicode escapes."""
    if isinstance(response, str):
        return clean_unicode_escapes(response)
    return response

def encode_image_to_base64(image_path):
    """
    Convert image to JPEG format and encode to base64 for smaller file size.
    Returns (encoded_string, media_type)
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (JPEG doesn't support alpha channel)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPEG to bytes with compression
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            image_data = buffer.getvalue()
            encoded = base64.b64encode(image_data).decode('utf-8')
            
            return encoded, 'image/jpeg'
            
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # Fallback to original method if conversion fails
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            encoded = base64.b64encode(image_data).decode('utf-8')
        
        # Try to detect format for fallback
        mime_type, _ = mimetypes.guess_type(image_path)
        detected_format = mime_type or 'image/jpeg'
        return encoded, detected_format



def encode_pil_image_to_base64(pil_image):
    """
    Convert PIL image to JPEG format and encode to base64 for smaller file size.
    """
    # Convert to RGB if necessary
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def list_transformed_pngs(dir_: Path) -> List[Path]:
    return sorted(
        p for p in dir_.iterdir()
        if p.is_file() and _IMG_RE.match(p.name)
    )


def resize_image_for_llama(image_path: str) -> str:
    """
    Resize image to meet Bedrock model requirements (1024x1024 max).
    
    Parameters
    ----------
    image_path : str
        Path to original image
        
    Returns
    -------
    str
        Path to resized image (or original if no resize needed)
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Bedrock models require images to fit within 1024x1024
            max_size = 1024
            
            # Check if resizing is needed
            if width <= max_size and height <= max_size:
                return image_path
            
            # Calculate new dimensions while maintaining aspect ratio
            ratio = min(max_size / width, max_size / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save to temporary file
            temp_dir = tempfile.gettempdir()
            temp_filename = f"resized_{os.path.basename(image_path)}"
            temp_path = os.path.join(temp_dir, temp_filename)
            resized_img.save(temp_path, "PNG")
            print(f"temp_path: {temp_path}")
            
            print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            return temp_path
            
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image_path  # Return original if resize fails