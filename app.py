import os
import streamlit as st
import urllib.parse
import pandas as pd
from dotenv import load_dotenv
from src.kgqa_engine import KGQAPipeline, GWDG_MODELS, GWDG_DEFAULT_MODEL

load_dotenv()

# Page config
st.set_page_config(
    page_title="KGQA — DBpedia Question Answering",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main .block-container { padding-top:2rem; padding-bottom:2rem; max-width:1200px; }

.kgqa-header {
  background: linear-gradient(135deg,#0f172a 0%,#1e3a5f 60%,#0e4d6b 100%);
  border-radius:16px; padding:2rem 2.5rem; margin-bottom:2rem;
  color:white; display:flex; align-items:center; gap:1.5rem;
  box-shadow:0 8px 32px rgba(0,0,0,0.3);
}
.kgqa-header h1 { margin:0; font-size:2rem; font-weight:700; letter-spacing:-0.5px; }
.kgqa-header p  { margin:0.25rem 0 0; opacity:0.75; font-size:0.9rem; }
.kgqa-badge {
  background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.2);
  border-radius:8px; padding:0.25rem 0.7rem; font-size:0.75rem;
  margin-top:0.4rem; display:inline-block;
}
.answer-card {
  background:linear-gradient(135deg,#f0fdf4,#dcfce7);
  border:1px solid #86efac; border-left:4px solid #16a34a;
  border-radius:12px; padding:1.25rem 1.5rem; margin:1rem 0;
}
.answer-card .label { font-size:0.75rem; font-weight:600; text-transform:uppercase;
  letter-spacing:0.5px; color:#15803d; margin-bottom:0.5rem; }
.answer-card .text  { font-size:1.15rem; font-weight:500; color:#14532d; line-height:1.5; }
.entity-chip {
  display:inline-block; background:#eff6ff; border:1px solid #bfdbfe;
  border-radius:20px; padding:0.3rem 0.8rem; margin:0.2rem;
  font-size:0.8rem; color:#1d4ed8; font-weight:500; text-decoration:none;
}
.badge-simple  { background:#dbeafe; color:#1e40af; border-radius:20px;
  padding:0.2rem 0.7rem; font-size:0.75rem; font-weight:600; border:1px solid #93c5fd; }
.badge-complex { background:#fef3c7; color:#92400e; border-radius:20px;
  padding:0.2rem 0.7rem; font-size:0.75rem; font-weight:600; border:1px solid #fcd34d; }
.sparql-box {
  background:#0f172a; color:#e2e8f0; border-radius:10px;
  padding:1rem 1.25rem; font-family:'Fira Code',monospace;
  font-size:0.8rem; line-height:1.6; overflow-x:auto;
  white-space:pre-wrap; border:1px solid #1e293b;
}
.metric-card {
  background:white; border:1px solid #e2e8f0; border-radius:10px;
  padding:1rem; text-align:center; box-shadow:0 1px 4px rgba(0,0,0,0.06);
}
.metric-card .val { font-size:1.5rem; font-weight:700; color:#1e3a5f; }
.metric-card .lbl { font-size:0.75rem; color:#64748b; margin-top:0.1rem; }
.history-item {
  background:white; border:1px solid #e2e8f0; border-radius:10px;
  padding:0.9rem 1.1rem; margin-bottom:0.75rem;
}
.history-item .q { font-weight:500; font-size:0.875rem; color:#1e293b; }
.history-item .a { font-size:0.8rem; color:#64748b; margin-top:0.3rem; }
#MainMenu, footer { visibility:hidden; }
.stDeployButton   { display:none; }
.stButton > button {
  background:linear-gradient(135deg,#1e3a5f,#0e4d6b) !important;
  color:white !important; border-radius:10px !important; border:none !important;
  padding:0.65rem 1.5rem !important; font-weight:600 !important; font-size:0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="kgqa-header">
  <div>
    <h1>KGQA — Knowledge Graph QA</h1>
    <p>Natural language questions answered from <strong>DBpedia</strong> and <strong>Wikidata</strong></p>
    <span class="kgqa-badge">Advanced Machine Learning · WiSe 2025/26 · Leuphana University</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Session state
if "history"     not in st.session_state: st.session_state.history     = []
if "last_result" not in st.session_state: st.session_state.last_result = None

# API key
def get_api_key() -> str:
    try:
        return st.secrets["GWDG_API_KEY"]
    except Exception:
        pass
    return os.getenv("GWDG_API_KEY", "")

# Sidebar
with st.sidebar:
    st.markdown("## Configuration")

    api_key = get_api_key()
    if api_key:
        st.caption("API key loaded")
    else:
        st.error("GWDG_API_KEY not found. Add it to your .env file.")

    model = st.selectbox(
        "LLM Model",
        options=GWDG_MODELS,
        index=GWDG_MODELS.index(GWDG_DEFAULT_MODEL),
        help="Llama 3.3 70B recommended for best SPARQL quality",
    )
    st.caption("[GWDG docs](https://docs.hpc.gwdg.de/services/saia/index.html) · [Models](https://docs.hpc.gwdg.de/services/chat-ai/models/index.html)")
    st.divider()

    st.markdown("### Example Questions")
    examples = {
        "Simple": [
            "Who is the architect of the Eiffel Tower?",
            "What is the capital of Germany?",
            "When was Albert Einstein born?",
            "Who founded Apple Inc.?",
            "What is the population of Paris?",
            "Who wrote Harry Potter?",
        ],
        "Complex": [
            "When was the architect of the Eiffel Tower born?",
            "What is the birthplace of the founder of Microsoft?",
            "What university did Barack Obama attend?",
        ],
        "Recent / Current": [
            "Who is the current president of France?",
            "Who is the current CEO of Tesla?",
            "Who won the Nobel Prize in Physics in 2024?",
        ],
        "Aggregation": [
            "How many films did Steven Spielberg direct?",
            "How many countries are in the European Union?",
        ],
    }
    selected_q = None
    for cat, qs in examples.items():
        with st.expander(cat, expanded=False):
            for q in qs:
                if st.button(q, key=f"ex_{q[:35]}", width='stretch'):
                    selected_q = q

    st.divider()
    st.markdown("### Session Stats")
    total = len(st.session_state.history)
    if total:
        avg_t = sum(h.get("duration_s", 0) for h in st.session_state.history) / total
        c1, c2 = st.columns(2)
        c1.metric("Questions", total)
        c2.metric("Avg Time", f"{avg_t:.1f}s")
        if st.button("Clear History", width='stretch'):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No questions asked yet.")

    st.divider()
    st.markdown("### Pipeline")
    st.markdown("""
```
Question
  -> Freebase ID Translation
  -> Entity Linking
  -> SPARQL Generation (LLM)
  -> DBpedia SPARQL
  -> Wikidata SPARQL
  -> Answer Synthesis (LLM)
```
""")

# Main input
col_in, col_btn = st.columns([5, 1])
with col_in:
    question = st.text_input(
        "question", label_visibility="collapsed",
        placeholder="e.g. Who is the architect of the Eiffel Tower?",
        value=selected_q or "", key="question_input",
    )
with col_btn:
    submit = st.button("Ask", width='stretch')

# Pipeline cache
@st.cache_resource(show_spinner=False)
def get_pipeline(key: str, mdl: str) -> KGQAPipeline:
    return KGQAPipeline(api_key=key, model=mdl)

# Run
if submit and question.strip():
    if not api_key:
        st.warning("No API key found. Add GWDG_API_KEY to your .env file.")
    else:
        pipeline = get_pipeline(api_key, model)
        with st.spinner("Linking entities, generating SPARQL, querying KG, synthesising answer..."):
            result = pipeline.answer(question.strip())
        st.session_state.last_result = result
        st.session_state.history.insert(0, result)
        st.rerun()

# Results
if st.session_state.last_result:
    r = st.session_state.last_result

    st.markdown(f"""
    <div class="answer-card">
      <div class="label">Answer</div>
      <div class="text">{r['answer']}</div>
    </div>
    """, unsafe_allow_html=True)

    kg       = r.get("kg_used", "dbpedia")
    fb_list  = r.get("freebase_resolved", [])
    fallback = r.get("fallback_used", False)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(f"""<div class="metric-card"><div class="val">{r['duration_s']}s</div><div class="lbl">Response Time</div></div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-card"><div class="val">{len(r['entities'])}</div><div class="lbl">Entities Linked</div></div>""", unsafe_allow_html=True)
    m3.markdown(f"""<div class="metric-card"><div class="val">{len(r['sparql_results'])}</div><div class="lbl">SPARQL Results</div></div>""", unsafe_allow_html=True)
    badge = '<span class="badge-complex">Complex</span>' if r["complexity"] == "complex" else '<span class="badge-simple">Simple</span>'
    m4.markdown(f"""<div class="metric-card"><div class="val" style="font-size:0.95rem;padding-top:0.3rem">{badge}</div><div class="lbl" style="margin-top:0.5rem">Complexity</div></div>""", unsafe_allow_html=True)
    fb_mark = " (fallback)" if fallback else ""
    m5.markdown(f"""<div class="metric-card"><div class="val" style="font-size:0.85rem">{kg.replace(' (fallback)','')+fb_mark}</div><div class="lbl" style="margin-top:0.4rem">KG Source</div></div>""", unsafe_allow_html=True)

    signals = []
    if fallback:
        signals.append(f"<b>Fallback used</b> — switched to <b>{kg}</b>")
    if fb_list:
        fb_str = ", ".join(f"<code>{e['freebase_id']}</code> -> <b>{e['label']}</b>" for e in fb_list)
        signals.append(f"<b>Freebase IDs resolved:</b> {fb_str}")
    if r.get("cleaned_question") != r.get("question"):
        signals.append(f"Question rewritten: <i>{r.get('cleaned_question','')}</i>")
    if signals:
        st.markdown(f"""<div style="background:#fefce8;border:1px solid #fde047;border-radius:10px;
            padding:0.65rem 1rem;font-size:0.82rem;color:#78350f;margin-top:0.75rem">
            {"  &nbsp;·&nbsp;  ".join(signals)}</div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    with st.expander("Technical Details", expanded=False):
        t1, t2, t3 = st.tabs(["Entity Linking", "SPARQL Query", "Raw Results"])

        with t1:
            ents = r.get("entities", [])
            if ents:
                chips = ""
                for e in ents:
                    uri   = e.get("uri", "#")
                    label = e.get("surface") or e.get("label") or uri.split("/")[-1]
                    short = uri.split("/")[-1]
                    is_wd = e.get("source") == "wikidata"
                    bg    = "#f0fdf4" if is_wd else "#eff6ff"
                    bdr   = "#86efac" if is_wd else "#bfdbfe"
                    clr   = "#166534" if is_wd else "#1d4ed8"
                    chips += f'<a href="{uri}" target="_blank" class="entity-chip" style="background:{bg};border-color:{bdr};color:{clr}">{label} -> {short}</a>'
                st.markdown(chips, unsafe_allow_html=True)
                st.caption("Blue = DBpedia   Green = Wikidata")
            else:
                st.info("No entities linked — SPARQL was inferred from the question text.")

        with t2:
            sparql_text = r.get("sparql", "")
            st.markdown(f'<div class="sparql-box">{sparql_text}</div>', unsafe_allow_html=True)
            if sparql_text:
                if "wikidata" in kg:
                    link     = "https://query.wikidata.org/#" + urllib.parse.quote(sparql_text)
                    label_ep = "Wikidata Query Service"
                else:
                    link     = "https://dbpedia.org/sparql?query=" + urllib.parse.quote(sparql_text)
                    label_ep = "DBpedia SPARQL Endpoint"
                st.markdown(f"[Run on {label_ep}]({link})")

        with t3:
            if r["sparql_results"]:
                rows = [{k: v.get("value", "") for k, v in b.items()} for b in r["sparql_results"][:20]]
                st.dataframe(pd.DataFrame(rows), width='stretch')
            else:
                st.info("No SPARQL results. Answer was synthesised from entity context.")

    st.divider()

# History
if len(st.session_state.history) > 1:
    st.markdown("### Recent Questions")
    for h in st.session_state.history[1:6]:
        complexity = "Complex" if h.get("complexity") == "complex" else "Simple"
        preview    = h.get("answer", "")[:120] + ("..." if len(h.get("answer", "")) > 120 else "")
        st.markdown(f"""
        <div class="history-item">
          <div class="q">[{complexity}] {h['question']}</div>
          <div class="a">{preview}</div>
        </div>""", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="margin-top:3rem;border:none;border-top:1px solid #e2e8f0"/>
<p style="text-align:center;color:#94a3b8;font-size:0.8rem;padding-bottom:1rem">
  KGQA System · DBpedia + Wikidata · Leuphana University · Advanced AI WiSe 2025/26
</p>
""", unsafe_allow_html=True)