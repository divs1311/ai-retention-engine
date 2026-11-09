"""Command-line interface: ``retention-engine --help``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from retention_engine import __version__, pipeline
from retention_engine.analytics.kpis import kpi_summary
from retention_engine.config import get_settings
from retention_engine.data.io import load_dataset

app = typer.Typer(
    help="Quick-commerce retention intelligence: cohorts, churn ML and LLM lifecycle CRM.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

ProviderOpt = typer.Option("auto", "--provider", "-p", help="auto | anthropic | openai | template")


@app.callback()
def _main(
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        console.print(f"retention-engine {__version__}")
        raise typer.Exit()


@app.command()
def generate(
    users: int = typer.Option(None, "--users", "-n", help="Number of simulated users."),
    seed: int = typer.Option(None, help="Random seed."),
) -> None:
    """Simulate a quick-commerce marketplace (users + orders)."""
    s = get_settings()
    if users:
        s.simulation.n_users = users
    if seed is not None:
        s.simulation.seed = seed
    u, o = pipeline.run_generate(s)
    console.print(f"[green]✓[/] {len(u):,} users, {len(o):,} orders → {s.data_dir}")


@app.command()
def kpis() -> None:
    """Print headline KPIs."""
    _, orders = load_dataset()
    k = kpi_summary(orders).to_dict()
    t = Table(title="Headline KPIs", show_header=False)
    for key, val in k.items():
        t.add_row(key.replace("_", " "), f"{val:,}" if isinstance(val, (int, float)) else str(val))
    console.print(t)


@app.command()
def train() -> None:
    """Train and evaluate the churn model (out-of-time validation)."""
    card = pipeline.run_train()
    t = Table(title=f"Churn model — selected: {card['selected_model']}")
    cols = ["roc_auc", "pr_auc", "brier", "ece", "lift_at_top10pct", "recall_at_top20pct"]
    t.add_column("model")
    for c in cols:
        t.add_column(c, justify="right")
    for name, m in card["metrics"].items():
        t.add_row(name, *[f"{m[c]:.3f}" for c in cols])
    console.print(t)
    b = card["baselines"]
    console.print(
        f"Same outreach budget as the 'inactive > 14d' rule: precision "
        f"{b['inactive_14d_rule']['precision']:.1%} → {b['model_at_same_budget']['precision']:.1%}, "
        f"recall {b['inactive_14d_rule']['recall']:.1%} → {b['model_at_same_budget']['recall']:.1%}"
    )


@app.command()
def score(top: int = typer.Option(10, help="Rows to preview.")) -> None:
    """Score the active user base and write the prioritised risk queue."""
    df = pipeline.run_score()
    t = Table(title=f"Top {top} users by value at risk")
    for c in ["user_id", "churn_p", "tier", "value_at_risk", "stage"]:
        t.add_column(c, no_wrap=True)
    t.add_column("why", overflow="ellipsis", no_wrap=True)
    for _, r in df.head(top).iterrows():
        t.add_row(
            str(r.user_id),
            f"{r.churn_probability:.2f}",
            r.risk_tier,
            f"{r.value_at_risk:,.0f}",
            r.lifecycle_stage,
            r.reason_text[:70],
        )
    console.print(t)


@app.command()
def campaigns(
    provider: str = ProviderOpt,
    limit: int = typer.Option(None, help="Only the N highest value-at-risk audiences."),
) -> None:
    """Generate lifecycle CRM campaigns for each audience."""
    out = pipeline.run_campaigns(provider=provider, limit=limit)
    for c in out[:5]:
        console.rule(f"{c.brief.audience_id}  ·  {c.brief.size} users  ·  {c.provider}")
        console.print(f"[bold]Push:[/] {c.content.push_title} — {c.content.push_body}")
        console.print(f"[bold]WhatsApp:[/] {c.content.whatsapp_message}")
        console.print(f"[bold]Email:[/] {c.content.email_subject}")
        status = "[green]passed[/]" if c.guardrails.passed else f"[red]{c.guardrails.violations}[/]"
        console.print(f"Guardrails: {status}")
    console.print(f"\n[green]✓[/] {len(out)} campaigns → {get_settings().campaigns_path}")


@app.command("run")
def run_all(provider: str = ProviderOpt) -> None:
    """Run the full pipeline: generate → train → score → campaigns."""
    out = pipeline.run_all(provider=provider)
    console.print(f"[green]✓ Pipeline complete[/] — {len(out)} campaigns generated")


@app.command()
def api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Serve the REST API (FastAPI + Uvicorn)."""
    import uvicorn

    uvicorn.run("retention_engine.api.main:app", host=host, port=port, reload=reload)


@app.command()
def dashboard(port: int = 8501) -> None:
    """Launch the Streamlit dashboard."""
    app_path = Path(__file__).resolve().parents[2] / "app" / "Home.py"
    raise typer.Exit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
        )
    )


if __name__ == "__main__":  # pragma: no cover
    app()
