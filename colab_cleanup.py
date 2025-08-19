# CLEANUP SCRIPT - Run this in Colab to free space

print("🧹 Cleaning up duplicate model downloads...")
print("=" * 60)

# Check disk usage
!df -h /

print("\n📊 What's using space:")
!du -sh /content/* 2>/dev/null | sort -hr | head -10

print("\n🔍 HuggingFace cache locations:")
!du -sh ~/.cache/huggingface/* 2>/dev/null | sort -hr
!du -sh /root/.cache/huggingface/* 2>/dev/null | sort -hr

print("\n🗑️ Cleaning up duplicates...")

# Clear HuggingFace cache duplicates
!rm -rf ~/.cache/huggingface/hub/models--*/blobs/*.incomplete
!rm -rf /root/.cache/huggingface/hub/models--*/blobs/*.incomplete

# Clear any temp downloads
!rm -rf /tmp/*.safetensors
!rm -rf /tmp/hf_*
!rm -rf /content/*.safetensors

# Clear PyTorch cache
import torch
torch.cuda.empty_cache()

# Show what models are actually downloaded
print("\n✅ Models currently in cache:")
!ls -la ~/.cache/huggingface/hub/ 2>/dev/null | grep models
!ls -la /root/.cache/huggingface/hub/ 2>/dev/null | grep models

print("\n📊 Space after cleanup:")
!df -h /

print("\n💡 To completely reset:")
print("Runtime -> Manage sessions -> TERMINATE")
print("This will clear everything and start fresh")