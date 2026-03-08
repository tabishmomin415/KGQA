# KGQA — Knowledge Graph Question Answering

> **Advanced Machine Learning · WiSe 2025/26 · Leuphana University of Lüneburg**
> Instructor: Prof. Debayan Banerjee & Kai Moltzen

A Knowledge Graph Question Answering (KGQA) system that takes natural language questions and answers them using the **DBpedia** and **Wikidata** Knowledge Graphs. Generates explainable SPARQL queries, handles simple and complex multi-hop questions, with a Streamlit web interface.

---

## System Architecture

```
graph TD
    %% Styling
    classDef engine fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef external fill:#e1f5fe,stroke:#01579b,stroke-width:2px,stroke-dasharray: 5 5;
    classDef database fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef ui fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    %% Nodes
    User((fa:fa-user User)) 
    UI[fa:fa-desktop Streamlit Web App]:::ui
    
    subgraph Core_Engine ["Core KGQA Engine (Python)"]
        direction TB
        Prep[Freebase ID Translation]:::engine
        EL[Entity Linking: DBpedia Spotlight]:::engine
        Gen[SPARQL Generation: Llama 3.3]:::engine
        Exec{fa:fa-microchip Execute Query}:::engine
    end

    Spotlight((fa:fa-cloud DBpedia API)):::external
    GWDG((fa:fa-bolt GWDG AI Service)):::external

    DB[(fa:fa-database DBpedia Endpoint)]:::database
    WD[(fa:fa-database Wikidata Fallback)]:::database
    Synth[fa:fa-comment Answer Synthesis]:::ui

    %% Connections
    User -->|Natural Language Question| UI
    UI --> Prep
    Prep --> EL
    EL <--> Spotlight
    EL --> Gen
    Gen <--> GWDG
    Gen --> Exec
    
    Exec -->|Primary| DB
    Exec -->|Empty Result| WD
    
    DB --> Synth
    WD --> Synth
    Synth -->|Clean Answer| UI
```

## Project Structure

```
KGQA
├── app.py                ← Streamlit web interface
├── requirements.txt      ← Python dependencies
├── .env                  ← Your GWDG API key (local only, gitignored)
├── .gitignore
├── questions.txt         ← Working / non-working questions for testing
├── README.md
└── src/
    ├── __init__.py       ← empty file
    └── kgqa_engine.py    ← core KGQA pipeline
```

---

## Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/tabishmomin415/KGQA.git
cd KGQA
```

### 2. Create virtual environment
```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your GWDG API key
Create a file called `.env` in the root folder with this content:
```
GWDG_API_KEY=your-actual-gwdg-key-here
```
Get your key at: https://saia.gwdg.de/dashboard

### 5. Run
```bash
streamlit run app.py
```
Opens at **http://localhost:8501** — the key loads automatically from `.env`.

---

## Deploy on Streamlit Cloud

1. Push this repo to GitHub (must be **public**)
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
3. Click **New app** → select repo → branch `main` → main file `app.py` → **Deploy**
4. Once deployed: app **⋮ menu** → **Settings** → **Secrets** → paste:
```toml
GWDG_API_KEY = "your-actual-gwdg-key-here"
```
5. Click Save — app restarts and key loads automatically.

---

## LLM — GWDG SAIA API

Uses the **GWDG academiccloud.de** OpenAI-compatible API for academic researchers.

- Endpoint: `https://chat-ai.academiccloud.de/v1`
- Docs: https://docs.hpc.gwdg.de/services/saia/index.html
- Default model: `llama-3.3-70b-instruct`

All models are selectable in the UI sidebar.

---

## Test Questions

See `questions.txt` for the full list of working and non-working questions.

**Quick tests:**

| Question | Expected Answer |
|----------|----------------|
| Who is the architect of the Eiffel Tower? | Stephen Sauvestre |
| What is the capital of Germany? | Berlin |
| When was Albert Einstein born? | 14 March 1879 |
| When was the architect of the Eiffel Tower born? | 26 December 1847 |
| Who is the current president of France? | Emmanuel Macron |

---

## Team

- Tabish Momin (4002968)
- Hassan Riaz Butt (4002898)
- Darshan Mehta (4004772)

---

## References

- DBpedia: https://dbpedia.org
- Wikidata: https://www.wikidata.org
- DBpedia Spotlight: https://www.dbpedia-spotlight.org
- LC-QuAD 2.0: https://github.com/AskNowQA/LC-QuAD
- GWDG SAIA: https://docs.hpc.gwdg.de/services/saia/index.html