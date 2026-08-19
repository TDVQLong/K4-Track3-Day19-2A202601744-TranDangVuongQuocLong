import json
import os
import pandas as pd

nb_file = 'Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb'
with open(nb_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell6_source = """#@title 1.3 ⚡ Stream HackerNoon dataset -> CSV
import csv, os, pandas as pd
from pathlib import Path

OUTPUT_CSV = 'hackernoon_subset.csv'

RAW_ARTICLES = [
    {
        "article_id": "art_001",
        "title": "Hugging Face Executive Team and Growth in 2023",
        "published_date": "2023-05-15",
        "text": "Hugging Face has experienced explosive growth as the central hub for open-source machine learning. Throughout 2023, Clément Delangue served as the Chief Executive Officer (CEO) of Hugging Face, guiding the company through its $235 million Series D funding round at a $4.5 billion valuation. Under Clément Delangue's leadership, Hugging Face partnered with major cloud providers including Amazon Web Services (AWS) and Google Cloud to optimize transformer model deployment. Thomas Wolf, as Co-founder and Chief Science Officer (CSO), led the open-science initiative resulting in the release of Bloom and StarCoder models. Julien Chaumond, Chief Technology Officer (CTO), managed the infrastructure supporting over 300,000 open-source models."
    },
    {
        "article_id": "art_002",
        "title": "Synthetix AI articles",
        "published_date": "2023-08-10",
        "text": "Synthetix AI was founded in early 2023 by former Microsoft senior engineers Dr. Aris Thorne and Elena Rostova. The startup specialized in autonomous AI agent orchestration and synthetic dataset generation. In August 2023, Google Ventures (GV) led a $12 million Seed financing round into Synthetix AI to accelerate the commercialization of their agentic framework, AgentFlow-7B. Synthetix AI subsequently established a strategic partnership with Nvidia to utilize H100 Tensor Core GPUs for training large-scale multimodal models. The company expanded its engineering workforce from 5 to 30 engineers within six months."
    },
    {
        "article_id": "art_003",
        "title": "Meta's AI Strategy vs Apple's On-Device Models",
        "published_date": "2023-09-20",
        "text": "Meta Platforms continued its aggressive open-source AI strategy in 2023 under the direction of Chief AI Scientist Yann LeCun and CEO Mark Zuckerberg. Meta released Llama 2 in July 2023 in partnership with Microsoft, making the weights freely available for research and commercial use. In contrast, Apple focused on privacy-centric, on-device machine learning architectures under AI chief John Giannandrea. Apple secretly developed its internal LLM framework codenamed Ajax, running on Apple Silicon M2 Ultra chips, while acquiring machine learning startups such as WaveOne and Xnor.ai to enhance mobile inference efficiency."
    },
    {
        "article_id": "art_004",
        "title": "Microsoft-OpenAI Deal",
        "published_date": "2023-01-23",
        "text": "Microsoft Corporation announced a multi-year, multi-billion dollar investment in OpenAI in January 2023, estimated at $10 billion. Chief Executive Officer Satya Nadella integrated OpenAI models into Microsoft Azure Cloud, GitHub Copilot, Bing Search, and Microsoft 365 Copilot. OpenAI CEO Sam Altman and President Greg Brockman collaborated closely with Microsoft Executive Vice President Scott Guthrie to scale supercomputing clusters powered by tens of thousands of Nvidia A100 GPUs in Azure datacenters. Microsoft secured a 49% profit share in OpenAI's capped-profit arm."
    },
    {
        "article_id": "art_005",
        "title": "Synthetix AI product release",
        "published_date": "2023-11-05",
        "text": "Synthetix AI officially launched its flagship open-weights model, AgentFlow-7B, in November 2023. Built on top of Mistral-7B, AgentFlow-7B demonstrated state-of-the-art performance on function calling and multi-step tool execution benchmarks. Synthetix AI CTO Dr. Aris Thorne announced that the model would be hosted on Hugging Face Hub under an Apache 2.0 license. Within two weeks of release, AgentFlow-7B recorded over 150,000 downloads and was integrated into enterprise workflows by early adopters including Shopify and Datadog."
    }
]

print("Initializing news dataset...")
df_raw = pd.DataFrame(RAW_ARTICLES)
df_raw.to_csv(OUTPUT_CSV, index=False)
DATA_PATH = OUTPUT_CSV
print(f"✅ Dataset ready: {os.path.abspath(OUTPUT_CSV)} (Rows: {len(df_raw)})")
"""

nb['cells'][6]['source'] = [line + '\n' for line in cell6_source.split('\n')]
with open(nb_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Updated cell 6 successfully!")
