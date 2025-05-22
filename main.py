#!/usr/bin/env python
"""
Local LangGraph agent backed by an Ollama-served LLM.
Run: python main.py --model llama3:8b-instruct \
                    --csv data/demo.csv \
                    --question "Which product line had highest 2024 Q1 margin?"
"""
from __future__ import annotations
import argparse, subprocess, sys, pathlib, textwrap, os
import pandas as pd
from typing import TypedDict, Dict, Any

from langchain_ollama import ChatOllama            # :contentReference[oaicite:2]{index=2}
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph             # :contentReference[oaicite:3]{index=3}

# ---------- state schema ----------------------------------------------------

class AgentState(TypedDict):
    model: str
    sys_prompt: str
    csv_dfs: Dict[str, pd.DataFrame]
    question: str
    answer: str

# ---------- helpers ---------------------------------------------------------

def ensure_model(model: str):
    """Pull model if missing and verify the daemon is up."""
    try:
        subprocess.run(["ollama", "show", model],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print(f"Pulling model {model} …")
        subprocess.run(["ollama", "pull", model], check=True)
    # quick health check
    ping = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if ping.returncode or model not in ping.stdout:
        sys.exit("❌  Ollama daemon not running (`ollama serve`).")

def load_csvs(paths: list[pathlib.Path]) -> dict[str, pd.DataFrame]:
    dfs = {}
    for p in paths:
        dfs[p.stem] = pd.read_csv(p)
    return dfs

# ---------- graph nodes -----------------------------------------------------

def agent_node(state):
    """Core reasoning node."""
    llm = ChatOllama(model=state["model"], temperature=0.0)
    df_tools = [create_pandas_dataframe_agent(llm, df, prefix=state["sys_prompt"], allow_dangerous_code=True)
                for df in state["csv_dfs"].values()]
    # Very simple: use *first* CSV agent; extend to tool-routing as you like
    response = df_tools[0].invoke(state["question"])
    # Extract the answer from the response dictionary
    answer = response.get("output", str(response))
    return {"answer": answer}

def final_node(state):
    """Format and print the answer."""
    answer = state["answer"]
    if isinstance(answer, dict):
        answer = str(answer)
    print("📝  Answer:\n", textwrap.fill(answer, width=100))
    return state

# ---------- wire up graph ----------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("final", final_node)
    graph.set_entry_point("agent").add_edge("agent", "final")
    return graph.compile()

# ---------- CLI --------------------------------------------------------------

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-r1:8b")
    ap.add_argument("--prompt", default="prompts/system_prompt.txt")
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--question", default="prompts/user_prompt.txt")
    args = ap.parse_args()

    ensure_model(args.model)
    csv_paths = [pathlib.Path(p) for p in args.csv]
    for p in csv_paths:
        if not p.exists():
            sys.exit(f"CSV not found: {p}")

    state = {
        "model": args.model,
        "sys_prompt": pathlib.Path(args.prompt).read_text(),
        "csv_dfs": load_csvs(csv_paths),
        "question": pathlib.Path(args.question).read_text(),
    }
    graph = build_graph()
    graph.invoke(state)          # use invoke() instead of run()

if __name__ == "__main__":
    cli()
