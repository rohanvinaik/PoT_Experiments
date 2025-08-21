import json
import hashlib
from pathlib import Path

models = [
    "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Base-Q4",
    "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Instruct-Q4"
]

results = {}
for model_path in models:
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        # Extract key info
        model_name = Path(model_path).name
        results[model_name] = {
            "architecture": config.get("architectures", ["unknown"])[0],
            "hidden_size": config.get("hidden_size", "unknown"),
            "num_layers": config.get("num_hidden_layers", "unknown"),
            "num_experts": config.get("num_local_experts", "unknown"),
            "vocab_size": config.get("vocab_size", "unknown"),
            "config_hash": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()[:16]
        }
        
        print(f"\n{model_name}:")
        for key, value in results[model_name].items():
            print(f"  {key}: {value}")

# Compare
if len(results) == 2:
    names = list(results.keys())
    if results[names[0]]["config_hash"] == results[names[1]]["config_hash"]:
        print("\n✅ Models have IDENTICAL configurations")
    else:
        print("\n❌ Models have DIFFERENT configurations")
        print("  This is expected for Base vs Instruct variants")

# Save results
import json
with open("$OUTPUT_DIR/config_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
