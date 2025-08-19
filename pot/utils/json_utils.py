import json
import numpy as np
from typing import Any, Dict, List, Union
from pathlib import Path
import warnings

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types properly"""
    
    def default(self, obj):
        # Handle numpy booleans
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle numpy integers
        if isinstance(obj, np.integer):
            return int(obj)
        
        # Handle numpy floats
        if isinstance(obj, np.floating):
            return float(obj)
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
        
        # Handle bytes
        if isinstance(obj, bytes):
            return obj.hex()
        
        # Let the default encoder handle it
        return super().default(obj)

def safe_json_dump(data: Dict[str, Any], filepath: Union[str, Path], **kwargs):
    """Safely dump data to JSON, handling numpy types"""
    filepath = Path(filepath)
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write with custom encoder
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyJSONEncoder, **kwargs)

def validate_no_mocks(data: Dict[str, Any]) -> bool:
    """Check if data contains mock placeholders"""
    
    def check_value(value):
        if isinstance(value, str):
            mock_indicators = ['mock', 'placeholder', 'dummy', 'fake', 'test_only']
            return not any(indicator in value.lower() for indicator in mock_indicators)
        elif isinstance(value, dict):
            return all(check_value(v) for v in value.values())
        elif isinstance(value, list):
            return all(check_value(v) for v in value)
        return True
    
    is_valid = check_value(data)
    
    if not is_valid:
        warnings.warn("Data appears to contain mock/placeholder values!")
    
    return is_valid

def separate_mock_files(data: Dict[str, Any], 
                       filepath: Union[str, Path],
                       force_real: bool = False) -> Path:
    """Save to appropriate file based on whether data is real or mock"""
    filepath = Path(filepath)
    
    if not force_real and not validate_no_mocks(data):
        # Save to mock file
        mock_filepath = filepath.parent / f"{filepath.stem}_mock{filepath.suffix}"
        warnings.warn(f"Saving to mock file: {mock_filepath}")
        safe_json_dump(data, mock_filepath, indent=2)
        return mock_filepath
    else:
        # Save to real file
        safe_json_dump(data, filepath, indent=2)
        return filepath