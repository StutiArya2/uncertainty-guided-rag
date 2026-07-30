# Uncertainty-Guided Reversible Evidence Compression RAG System

A Retrieval-Augmented Generation (RAG) pipeline that compresses retrieved evidence in an
uncertainty-aware, *reversible* manner — reducing token load while preserving the ability
to restore evidence when claim support is insufficient.

## Pipeline Overview

1. **User Query** → submitted by the user
2. **Retrieval** → pulls candidate documents from the Knowledge Base (KB)
3. **Evidence Mapping** → maps retrieved evidence to individual claims (claim-wise)
4. **Initial Evidence Set** → full, high-token evidence set per claim
5. **Uncertainty-Guided Compression** → compresses evidence based on uncertainty estimates, reducing token count
6. **Claim Support Evaluation** → checks whether compressed evidence is sufficient to support each claim
   - **Sufficient** → proceed to Output Generation
   - **Insufficient** →
     6.1. **Evidence Restoration** → reverses compression to recover fuller evidence
     6.2. **Re-evaluation** → re-checks claim support with restored evidence
       - **Sufficient** → Output Generation
       - **Still insufficient** → **Abstain / Clarify / Retrieve More**
7. **Output Generation** → final answer generation once evidence is sufficient

## Project Status

- Target milestone: Mid-August (initial pipeline / dead-line outcome)
- Longer-term goal: Fall '26–'27 mini-project extension, targeting conference/journal submission (indexed publication)

## Tech Stack

- Python
- (RAG framework / vector store / LLM APIs — TBD)

## Repository Structure
uncertainty-guided-rag/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── src/
│ ├── retrieval.py
│ ├── evidence_mapping.py
│ ├── compression.py
│ ├── evaluation.py
│ ├── restoration.py
│ └── generation.py
├── data/
│ └── kb/
├── notebooks/
└── tests/

## Setup

```bash
git clone https://github.com/<your-username>/uncertainty-guided-rag.git
cd uncertainty-guided-rag
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Collaborators
