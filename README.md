# 💼 Job Application Assistant — Deep Agent

> Automatically finds 5 live job postings matching your profile and generates a tailored cover letter for each one — powered by **LangChain Deep Agents**, **Tavily Search**, and **OpenAI GPT-4o-mini**.

---

## 🏗️ Architecture

```
User Resume + Preferences
         │
         ▼
  ┌─────────────────────────────────────┐
  │        Deep Agent (Orchestrator)    │
  │  ─ Plans tasks with todo_write      │
  │  ─ Delegates to sub-agents          │
  │  ─ Maintains virtual file system    │
  └─────────┬──────────────┬────────────┘
            │              │
     ┌──────▼──────┐  ┌────▼──────────────┐
     │ Job Search  │  │ Cover Letter      │
     │  Sub-Agent  │  │  Writer Sub-Agent │
     │ (Tavily)    │  │  (GPT-4o-mini)    │
     └─────────────┘  └───────────────────┘
            │              │
            ▼              ▼
     5 Live Job URLs   cover_letters.md
            │              │
            └──────┬───────┘
                   ▼
          Streamlit UI (Table + DOCX download)
```

---

## ⚡ Quick Start

### 1. Clone / unzip the project

```bash
cd job_assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get API keys

| Service | Free tier | Link |
|---------|-----------|------|
| **OpenAI** | Pay-as-you-go (~$0.01 per run with gpt-4o-mini) | [platform.openai.com](https://platform.openai.com) |
| **Tavily** | 1,000 free credits/month | [app.tavily.com](https://app.tavily.com) |

### 4. Run the app

```bash
streamlit run app.py
```

The browser opens at `http://localhost:8501`. Paste your API keys in the sidebar, upload your resume, and click **Run Agent**.

---

## 🔧 Optional: Set keys via environment variables

```bash
export OPENAI_API_KEY="sk-proj-..."
export TAVILY_API_KEY="tvly-dev-..."
streamlit run app.py
```

If env vars are set, the sidebar fields will be pre-filled.

---

## 📁 Project Structure

```
job_assistant/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── .streamlit/
    └── config.toml     ← Optional Streamlit theme config
```

---

## 🎛️ Features

- **Resume parsing** — PDF, DOCX, and TXT formats supported
- **Deep Agent architecture** — Planning + sub-agent delegation
- **Live job search** — Tavily fetches real postings (not hallucinated)
- **Tailored cover letters** — Each letter references your specific skills & the job requirements
- **DOCX export** — Download all cover letters as a single Word document
- **Dark-themed UI** — GitHub-inspired dark theme with clean typography

---

## 🧠 How Deep Agents Work

Unlike a simple LLM chat, a Deep Agent:

1. **Plans** — Creates a to-do list before acting
2. **Delegates** — Spawns specialised sub-agents for focused tasks
3. **Uses a file system** — Sub-agents write to shared virtual files
4. **Manages context** — Each sub-agent has its own isolated context window

This is the same architecture used by Claude Code, OpenAI Deep Research, and Manus.

---

## 🛠️ Customisation

| What to change | Where |
|----------------|-------|
| LLM model | Sidebar dropdown (gpt-4o-mini / gpt-4o / gpt-4-turbo) |
| Number of jobs | Edit `return normed[:5]` in `normalize_jobs()` |
| Cover letter length | Edit `≤150 words` in `INSTRUCTIONS` |
| Add more tools | Add `@tool` functions and pass to `build_agent()` |
| Job sources | Edit the `INSTRUCTIONS` string to prefer specific sites |

---

## 🚨 Troubleshooting

| Problem | Fix |
|---------|-----|
| `No jobs found` | Broaden your title/location; check Tavily key |
| `JSON parse failed` | Retry — model occasionally formats JSON incorrectly |
| `Rate limit` | Wait 60 s or upgrade OpenAI tier |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Agent takes >3 min | Switch to `gpt-4o` for faster reasoning |

---

## 📜 Credits

Based on the tutorial by **Aashi Dutt** on [DataCamp](https://www.datacamp.com/tutorial/deep-agents).  
Architecture inspired by [LangChain Deep Agents](https://blog.langchain.com/deep-agents/).
