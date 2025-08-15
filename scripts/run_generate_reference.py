#!/usr/bin/env python
"""
Create and store reference fingerprints for a given model and challenge set.
Outputs JSONL with: challenge_id, family, items_hash, fingerprint_type, fingerprint_value, metadata.
"""

import argparse
import yaml
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pot.core.challenge import ChallengeConfig, generate_challenges
from pot.core.fingerprint import io_hash
from pot.core.logging import StructuredLogger
from pot.core.governance import new_session_nonce

def main():
    parser = argparse.ArgumentParser(description="Generate reference fingerprints")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    exp_name = config['experiment']
    logger = StructuredLogger(f"{args.output_dir}/{exp_name}/reference")
    
    # Generate challenges
    master_key = "0" * 64  # Placeholder master key
    session_nonce = new_session_nonce()
    
    for challenge_family in config['challenges']['families']:
        cfg = ChallengeConfig(
            master_key_hex=master_key,
            session_nonce_hex=session_nonce,
            n=challenge_family['n'],
            family=challenge_family['family'],
            params=challenge_family['params']
        )
        
        challenges = generate_challenges(cfg)
        
        # Log reference
        entry = {
            "challenge_id": challenges['challenge_id'],
            "family": challenges['family'],
            "n": challenge_family['n'],
            "items_hash": io_hash(challenges['items']),
            "fingerprint_type": "io_hash",
            "salt": challenges['salt'],
            "metadata": {
                "config": args.config,
                "session_nonce": session_nonce
            }
        }
        
        logger.log_jsonl("reference.jsonl", entry)
        print(f"Generated reference for {challenge_family['family']} with {challenge_family['n']} items")
    
    print(f"References saved to {args.output_dir}/{exp_name}/reference/")

if __name__ == "__main__":
    main()