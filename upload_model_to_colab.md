# How to Upload Mistral Model to Colab

## Option 1: Via Google Drive (Recommended)

### Step 1: Upload to Google Drive (from your computer)
1. Go to [Google Drive](https://drive.google.com)
2. Create a folder called `models`
3. Upload your Mistral model files from:
   - Mac: `~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/`
   - Look for files like: `*.safetensors`, `config.json`, `tokenizer.json`

### Step 2: Mount Drive in Colab
```python
# In Colab, run this:
from google.colab import drive
drive.mount('/content/drive')

# Check if files are there
!ls -la /content/drive/MyDrive/models/
```

### Step 3: Load from Drive
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load from your uploaded files
model_path = "/content/drive/MyDrive/models/mistral-7b/"
mistral = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,  # Don't try to download
    torch_dtype=torch.float16,
    device_map="auto"
)
mistral_tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
```

## Option 2: Direct Upload to Colab (For smaller files)

### In Colab:
```python
# Upload files directly
from google.colab import files

# This opens a file picker
uploaded = files.upload()

# Save to local Colab storage
import os
os.makedirs("/content/mistral_model", exist_ok=True)
for filename, content in uploaded.items():
    with open(f"/content/mistral_model/{filename}", "wb") as f:
        f.write(content)
```

## Option 3: Copy from HuggingFace Cache (if partially downloaded)

### Find what you have locally:
```bash
# On your Mac, check what's cached:
ls -la ~/.cache/huggingface/hub/

# Look for folders like:
# models--mistralai--Mistral-7B-Instruct-v0.3/
# models--HuggingFaceH4--zephyr-7b-beta/
```

### Zip the model folder:
```bash
# On your Mac
cd ~/.cache/huggingface/hub/
zip -r mistral_model.zip models--mistralai--Mistral-7B-Instruct-v0.3/
# This creates mistral_model.zip (~13GB)
```

### Upload the zip to Google Drive, then in Colab:
```python
# Mount drive
from google.colab import drive
drive.mount('/content/drive')

# Unzip
!unzip /content/drive/MyDrive/mistral_model.zip -d /content/

# Set HuggingFace cache to use these files
import os
os.environ['HF_HOME'] = '/content/'
os.environ['TRANSFORMERS_CACHE'] = '/content/'

# Now load normally - it will find the cached files
from transformers import AutoModelForCausalLM
mistral = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    local_files_only=True,  # Use cache only
    torch_dtype=torch.float16,
    device_map="auto"
)
```

## Option 4: Use Colab's Persistent Storage (Colab Pro)

If you have Colab Pro, you can save models persistently:
```python
# Save model to persistent storage
model.save_pretrained("/content/drive/MyDrive/models/mistral/")
tokenizer.save_pretrained("/content/drive/MyDrive/models/mistral/")

# Next session, just load:
model = AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/models/mistral/")
```