#!/usr/bin/env python3
"""
Run PoT Test Suite in Google Colab - Fixed Paths Version
Handles file paths correctly when copying from Google Drive
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

print("🚀 POT TEST SUITE FOR GOOGLE COLAB")
print("=" * 70)
print(f"Started at: {datetime.now()}")
print("=" * 70)

# Mount Google Drive
print("\n📁 Setting up environment...")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
    print("✅ Running in Google Colab")
except:
    IN_COLAB = False
    print("📍 Running locally")

# Setup paths
if IN_COLAB:
    work_dir = '/content/PoT_Experiments'
    source_dir = '/content/drive/MyDrive/pot_to_upload'
    
    # Check source and copy
    if os.path.exists(source_dir):
        print(f"📥 Copying from: {source_dir}")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        shutil.copytree(source_dir, work_dir)
        print("✅ Files copied")
    elif not os.path.exists(work_dir):
        print("❌ No source found. Upload your code to:")
        print("   /My Drive/pot_to_upload/")
        sys.exit(1)
    
    os.chdir(work_dir)
else:
    work_dir = os.getcwd()

print(f"📂 Working directory: {work_dir}")

# List what we have
print("\n📋 Checking available files...")
print("\nDirectories:")
for item in os.listdir('.'):
    if os.path.isdir(item) and not item.startswith('.'):
        print(f"  📁 {item}/")

# Check scripts directory specifically
if os.path.exists('scripts'):
    print("\nScripts directory contents:")
    scripts_files = os.listdir('scripts')
    for f in sorted(scripts_files)[:15]:
        path = os.path.join('scripts', f)
        if os.path.isfile(path):
            # Check if executable
            is_exec = os.access(path, os.X_OK)
            exec_mark = "✓" if is_exec else " "
            size = os.path.getsize(path)
            print(f"  [{exec_mark}] {f} ({size:,} bytes)")

# Install dependencies
print("\n📦 Installing dependencies...")
if os.path.exists('requirements.txt'):
    subprocess.run(['pip', 'install', '-q', '-r', 'requirements.txt'], check=False)
else:
    # Install essential packages
    subprocess.run(['pip', 'install', '-q', 'torch', 'transformers', 'numpy', 
                    'scipy', 'pandas', 'matplotlib', 'tabulate'], check=False)
print("✅ Dependencies installed")

# Set Python path
sys.path.insert(0, work_dir)
os.environ['PYTHONPATH'] = work_dir

# Create results directory
results_dir = 'experimental_results'
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("\n" + "="*70)
print("🧪 RUNNING TEST SUITE")
print("=" * 70)

# Define test scripts to run
test_scripts = [
    # Shell scripts
    ('scripts/run_all_quick.sh', 'bash', 'Quick validation tests'),
    ('scripts/run_all_fast.sh', 'bash', 'Fast test suite'),
    ('scripts/run_all.sh', 'bash', 'Full test suite'),
    ('scripts/run_standard_validation.sh', 'bash', 'Standard validation'),
    ('scripts/run_reliable_validation.sh', 'bash', 'Reliable validation'),
    
    # Python scripts
    ('scripts/test_llm_verification.py', 'python', 'LLM verification'),
    ('experimental_results/reliable_validation.py', 'python', 'Deterministic validation'),
    ('scripts/generate_validation_report.py', 'python', 'Generate report'),
]

results_summary = []

for script_path, interpreter, description in test_scripts:
    if os.path.exists(script_path):
        print(f"\n### {description} ###")
        print(f"📄 Script: {script_path}")
        
        # Make shell scripts executable
        if interpreter == 'bash' and os.path.exists(script_path):
            os.chmod(script_path, 0o755)
        
        try:
            # Construct command
            if interpreter == 'bash':
                cmd = ['bash', script_path]
            else:
                cmd = [sys.executable, script_path]
            
            # Run with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=work_dir,
                env={**os.environ, 'PYTHON': sys.executable}
            )
            
            # Save output
            output_name = os.path.basename(script_path).replace('.', '_')
            output_file = f"{results_dir}/{output_name}_{timestamp}.txt"
            with open(output_file, 'w') as f:
                f.write(f"Script: {script_path}\n")
                f.write(f"Exit code: {result.returncode}\n")
                f.write("\n=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            
            # Check result
            if result.returncode == 0:
                print(f"  ✅ PASSED")
                results_summary.append((description, 'PASSED'))
                
                # Show key output lines
                for line in result.stdout.split('\n')[-5:]:
                    if line.strip() and any(word in line.lower() for word in ['success', 'pass', 'complete', '✓', '✅']):
                        print(f"     {line.strip()}")
            else:
                print(f"  ⚠️ Exit code: {result.returncode}")
                results_summary.append((description, f'EXIT_{result.returncode}'))
                
                # Show errors if any
                if result.stderr:
                    err_lines = result.stderr.strip().split('\n')
                    for line in err_lines[-3:]:
                        if line.strip():
                            print(f"     Error: {line.strip()}")
                            
        except subprocess.TimeoutExpired:
            print(f"  ⏱️ TIMEOUT (>5 minutes)")
            results_summary.append((description, 'TIMEOUT'))
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results_summary.append((description, 'ERROR'))
    else:
        print(f"\n⚠️ Skipping {description} - not found: {script_path}")
        results_summary.append((description, 'NOT_FOUND'))

# Look for generated results
print("\n" + "="*70)
print("📊 COLLECTING RESULTS")
print("=" * 70)

# Find all JSON result files
import glob
json_files = glob.glob('**/*.json', recursive=True)
json_results = []

for json_file in json_files:
    # Skip unrelated files
    if any(skip in json_file for skip in ['node_modules', '.git', 'package', 'config']):
        continue
    
    # Check if it's a result file
    if any(pattern in json_file for pattern in ['result', 'validation', 'test', 'output']):
        size = os.path.getsize(json_file)
        if size > 100:  # More than 100 bytes (not empty)
            json_results.append(json_file)
            print(f"  ✅ Found: {json_file} ({size:,} bytes)")

# Check for markdown reports
md_reports = glob.glob('**/*report*.md', recursive=True)
for report in md_reports:
    if os.path.getsize(report) > 100:
        print(f"  📄 Report: {report}")

# Create summary
print("\n📋 Creating summary...")
summary = f"""# PoT Test Suite Execution Summary

**Date**: {datetime.now()}
**Environment**: {'Google Colab' if IN_COLAB else 'Local'}
**Working Directory**: {work_dir}

## Test Results

| Test | Status |
|------|--------|
"""

for desc, status in results_summary:
    emoji = "✅" if status == "PASSED" else "⚠️" if "EXIT" in status else "❌"
    summary += f"| {desc} | {emoji} {status} |\n"

summary += f"""

## Files Generated

- Result JSON files: {len(json_results)}
- Report files: {len(md_reports)}
- Output logs: {len(os.listdir(results_dir))}

## Result Files
"""

for json_file in json_results[:10]:
    summary += f"- `{json_file}`\n"

with open(f'{results_dir}/execution_summary_{timestamp}.md', 'w') as f:
    f.write(summary)

print("✅ Summary created")

# Create downloadable package
if IN_COLAB:
    print("\n📦 Creating results package...")
    import zipfile
    
    zip_path = f'/content/pot_results_{timestamp}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all results
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, work_dir)
                zipf.write(file_path, arcname)
        
        # Add JSON results
        for json_file in json_results:
            zipf.write(json_file, json_file)
        
        # Add reports
        for report in md_reports:
            zipf.write(report, report)
    
    size_mb = os.path.getsize(zip_path) / (1024*1024)
    print(f"✅ Package created: {zip_path} ({size_mb:.2f} MB)")
    
    # Download
    try:
        from google.colab import files
        files.download(zip_path)
        print("📥 Downloading...")
    except:
        print(f"📥 Download from: {zip_path}")

# Final report
print("\n" + "="*70)
print("🎉 EXECUTION COMPLETE!")
print("=" * 70)

print("\n📊 Summary:")
passed = sum(1 for _, status in results_summary if status == 'PASSED')
total = len(results_summary)
print(f"  • Tests run: {total}")
print(f"  • Passed: {passed}")
print(f"  • Success rate: {passed/total*100:.1f}%" if total > 0 else "  • No tests run")

if json_results:
    print(f"\n✅ Found {len(json_results)} result files with test data")
    print("📄 Check the downloaded package for full results")
else:
    print("\n⚠️ No JSON result files found")
    print("Check the output logs for details")

print("\n💡 If scripts weren't found:")
print("  1. Ensure all files uploaded to Google Drive")
print("  2. Check that directory structure is preserved")
print("  3. Verify scripts/ directory contains the .sh files")