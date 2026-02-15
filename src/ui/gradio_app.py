"""Gradio chat interface for ArXiv Paper Curator."""

import gradio as gr
import httpx

API_URL = "http://localhost:8000"


def ask_question(
    question: str,
    model: str,
    top_k: int,
    date_filter: int | None,
    categories: str,
) -> tuple[str, str]:
    """Send question to the API and format the response.

    Args:
        question: Natural-language research question.
        model: LLM model name (e.g. llama3, mistral).
        top_k: Number of retrieval results.
        date_filter: Limit to papers from last N days (0 = all).
        categories: Comma-separated arXiv categories.

    Returns:
        Tuple of (answer markdown, sources markdown).
    """
    if not question.strip():
        return "Please enter a question.", ""

    payload: dict = {
        "question": question,
        "model": model,
        "top_k": int(top_k),
    }
    if date_filter and date_filter > 0:
        payload["date_filter_days"] = int(date_filter)
    if categories.strip():
        payload["categories"] = [c.strip() for c in categories.split(",")]

    try:
        response = httpx.post(f"{API_URL}/api/ask", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        # Format answer with timing metadata
        answer = data["answer"]
        timing = (
            f"\n\n---\n*Retrieval: {data['retrieval_time_ms']}ms | "
            f"Generation: {data['generation_time_ms']}ms | "
            f"Total: {data['total_time_ms']}ms | Model: {data['model_used']}*"
        )

        # Format sources as markdown
        sources_md = "### Source Documents\n\n"
        for i, src in enumerate(data.get("sources", []), 1):
            sources_md += (
                f"**{i}. [{src['paper_title']}]({src['arxiv_url']})**\n"
                f"- arXiv: `{src['arxiv_id']}`\n"
                f"- Section: {src.get('section', 'N/A')}\n"
                f"- Relevance: {src['relevance_score']:.4f}\n"
                f"- Snippet: _{src['snippet']}_\n\n"
            )

        return answer + timing, sources_md

    except httpx.ConnectError:
        return (
            "**Connection Error:** Cannot reach the API server. "
            "Make sure FastAPI is running on http://localhost:8000\n\n"
            "Start it with: `uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000`",
            "",
        )
    except httpx.HTTPStatusError as e:
        return f"**API Error ({e.response.status_code}):** {e.response.text}", ""
    except Exception as e:
        return f"**Error:** {e!s}", ""


def check_health() -> str:
    """Check API health status."""
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        status_parts = [f"**API:** {data.get('status', 'unknown')}"]

        services = data.get("services", {})
        for svc, info in services.items():
            svc_status = info.get("status", "unknown") if isinstance(info, dict) else info
            status_parts.append(f"**{svc}:** {svc_status}")

        return " | ".join(status_parts)
    except Exception as e:
        return f"**Health check failed:** {e!s}"


# ── Build the UI ──────────────────────────────────────────────────
_CUSTOM_CSS = """
    .main-header { text-align: center; margin-bottom: 1rem; }
    .status-bar { font-size: 0.85rem; padding: 0.5rem; }
"""

with gr.Blocks() as demo:
    gr.Markdown(
        "# ArXiv Paper Curator\n" "### Ask questions about the latest AI research papers",
        elem_classes=["main-header"],
    )

    # Health bar
    with gr.Row():
        health_output = gr.Markdown("Click 'Check Status' to verify API connectivity.")
        health_btn = gr.Button("Check Status", size="sm")
    health_btn.click(fn=check_health, inputs=[], outputs=[health_output])

    gr.Markdown("---")

    with gr.Row():
        # Left column: question input
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What are the latest advances in RAG systems?",
                lines=3,
            )
            submit_btn = gr.Button("Ask", variant="primary")

        # Right column: settings
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                label="LLM Model",
                choices=["llama3", "mistral", "phi3", "gemma2"],
                value="llama3",
            )
            top_k_slider = gr.Slider(
                label="Top-K Results",
                minimum=1,
                maximum=20,
                value=5,
                step=1,
            )
            date_filter = gr.Number(
                label="Date Filter (last N days, 0 = all)",
                value=0,
            )
            categories_input = gr.Textbox(
                label="Categories (comma-separated)",
                placeholder="cs.AI, cs.LG",
                value="",
            )

    with gr.Row():
        with gr.Column(scale=2):
            answer_output = gr.Markdown(label="Answer")
        with gr.Column(scale=1):
            sources_output = gr.Markdown(label="Sources")

    # Wire up the submit action
    submit_btn.click(
        fn=ask_question,
        inputs=[question_input, model_dropdown, top_k_slider, date_filter, categories_input],
        outputs=[answer_output, sources_output],
    )

    # Also allow Enter key from question box
    question_input.submit(
        fn=ask_question,
        inputs=[question_input, model_dropdown, top_k_slider, date_filter, categories_input],
        outputs=[answer_output, sources_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=_CUSTOM_CSS,
    )
