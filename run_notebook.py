import os
import sys
import json
import nbformat
from nbclient import NotebookClient
from dotenv import load_dotenv

# 1. Load environment variables from .env
load_dotenv(override=True)

notebook_path = "Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb"
output_path = "Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb"

print(f"Reading notebook: {notebook_path}...")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# 2. Setup NotebookClient
client = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3",
    allow_errors=True,
    record_timing=True
)

print("Executing notebook cells...")
try:
    client.execute()
    print("Execution completed successfully!")
except Exception as e:
    print(f"Execution finished with warning/error: {e}")

# 3. Save executed notebook with all outputs populated
print(f"Saving executed notebook to {output_path}...")
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook outputs saved successfully!")
