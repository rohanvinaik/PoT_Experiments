#!/usr/bin/env python3
"""
FIND RESULTS - Show exactly where test results are saved in Colab
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("FINDING POT TEST RESULTS IN GOOGLE COLAB")
print("=" * 70)

def show_directory_contents(path, max_depth=2, current_depth=0):
    """Show directory contents with size info"""
    if current_depth > max_depth:
        return
        
    try:
        if not os.path.exists(path):
            print(f"❌ {path} does not exist")
            return
            
        print(f"\n📁 Contents of {path}:")
        items = os.listdir(path)
        if not items:
            print("  (empty directory)")
            return
            
        for item in sorted(items):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                file_count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
                print(f"  📁 {item}/ ({file_count} files)")
                if current_depth < max_depth:
                    show_directory_contents(item_path, max_depth, current_depth + 1)
            else:
                size = os.path.getsize(item_path)
                if size > 1024*1024:
                    size_str = f"{size/(1024*1024):.1f} MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size} bytes"
                print(f"  📄 {item} ({size_str})")
                
    except PermissionError:
        print(f"❌ Permission denied: {path}")
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")

# Check all possible locations
locations_to_check = [
    "/content",
    "/content/PoT_Experiments",
    "/content/PoT_Experiments/experimental_results",
    "/content/PoT_Experiments/test_results", 
    "/content/PoT_Results",
    "/content/drive/MyDrive"
]

print("\n🔍 Checking all possible result locations:")

for location in locations_to_check:
    if os.path.exists(location):
        print(f"\n✅ {location} exists")
        show_directory_contents(location, max_depth=1)
    else:
        print(f"\n❌ {location} does not exist")

# Look for any files with "result" in the name
print("\n" + "=" * 70)
print("🔍 SEARCHING FOR FILES WITH 'result' IN NAME")
print("=" * 70)

def find_result_files(search_path):
    """Find all files containing 'result' in name"""
    result_files = []
    try:
        for root, dirs, files in os.walk(search_path):
            for file in files:
                if 'result' in file.lower():
                    full_path = os.path.join(root, file)
                    size = os.path.getsize(full_path)
                    result_files.append((full_path, size))
    except:
        pass
    return result_files

# Search in /content
content_results = find_result_files("/content")
if content_results:
    print("\n📊 Files with 'result' in name:")
    for filepath, size in sorted(content_results):
        if size > 1024:
            size_str = f"{size/1024:.1f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"  {filepath} ({size_str})")
else:
    print("\n❌ No files with 'result' in name found in /content")

# Look for JSON files specifically
print("\n" + "=" * 70)
print("🔍 SEARCHING FOR JSON FILES")
print("=" * 70)

def find_json_files(search_path):
    """Find all JSON files"""
    json_files = []
    try:
        for root, dirs, files in os.walk(search_path):
            for file in files:
                if file.endswith('.json'):
                    full_path = os.path.join(root, file)
                    size = os.path.getsize(full_path)
                    json_files.append((full_path, size))
    except:
        pass
    return json_files

json_results = find_json_files("/content")
if json_results:
    print("\n📊 JSON files found:")
    for filepath, size in sorted(json_results):
        size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} bytes"
        print(f"  {filepath} ({size_str})")
        
        # Try to peek at JSON content
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    keys = list(data.keys())[:5]  # First 5 keys
                    print(f"    Keys: {keys}")
        except:
            print(f"    (could not read JSON)")
else:
    print("\n❌ No JSON files found in /content")

# Check if we're actually in the right directory
print("\n" + "=" * 70)
print("🔍 CURRENT WORKING DIRECTORY")
print("=" * 70)
print(f"Current directory: {os.getcwd()}")
if os.path.exists('scripts/run_all.sh'):
    print("✅ Found scripts/run_all.sh - we're in the right place")
else:
    print("❌ No scripts/run_all.sh - we might be in wrong directory")

# Show what's in current directory
show_directory_contents(os.getcwd(), max_depth=1)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("If you don't see results above, it means:")
print("1. Tests haven't run yet, OR")
print("2. Tests failed before creating results, OR") 
print("3. Results are in a different location")
print("\nTo run tests and see results, use:")
print("  !python simple_colab_runner.py")
print("=" * 70)