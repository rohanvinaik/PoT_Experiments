import json
import time
from pathlib import Path
from typing import Dict, Any

class StructuredLogger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def log_jsonl(self, filename: str, entry: Dict[str, Any]):
        filepath = self.output_dir / filename
        with open(filepath, 'a') as f:
            json.dump(entry, f)
            f.write('\n')