import json
from pathlib import Path
from datetime import datetime

output_dir = Path("$OUTPUT_DIR")
report = {
    "timestamp": datetime.now().isoformat(),
    "models": {
        "base": "Mixtral-8x22B-Base-Q4",
        "instruct": "Mixtral-8x22B-Instruct-Q4"
    },
    "tests_completed": [],
    "resource_limits": {
        "cpu_threads": 8,
        "memory_target_gb": 50,
        "nice_priority": 15
    }
}

# Check which tests completed
if (output_dir / "config_comparison.json").exists():
    report["tests_completed"].append("config_comparison")
    with open(output_dir / "config_comparison.json") as f:
        report["config_comparison"] = json.load(f)

if (output_dir / "statistical_result.json").exists():
    report["tests_completed"].append("statistical_verification")
    with open(output_dir / "statistical_result.json") as f:
        report["statistical_result"] = json.load(f)

# Generate summary
print("\n" + "=" * 60)
print("MIXTRAL MODEL VERIFICATION SUMMARY")
print("=" * 60)
print(f"Timestamp: {report['timestamp']}")
print(f"Models tested:")
print(f"  - Base: {report['models']['base']}")
print(f"  - Instruct: {report['models']['instruct']}")
print(f"\nTests completed: {', '.join(report['tests_completed'])}")

if "config_comparison" in report:
    print("\nConfiguration Analysis:")
    for model, info in report["config_comparison"].items():
        print(f"  {model}: {info.get('architecture', 'unknown')} architecture")

if "statistical_result" in report:
    print(f"\nStatistical Verification:")
    print(f"  Decision: {report['statistical_result']['decision']}")
    print(f"  Confidence: {report['statistical_result']['confidence']:.2%}")

print("\n" + "=" * 60)

# Save full report
with open(output_dir / "final_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n✅ Full report saved to: {output_dir}/final_report.json")
