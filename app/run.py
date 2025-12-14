from datetime import datetime, timezone
import json
from pathlib import Path

import httpx
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel

#----------------------
# Config models
#---------------------

class RunConfig(BaseModel):
	model: str="llama3.1:8b"
	temperature: float = 0.2
	prompt: str

class RunResult(BaseModel):
	config: RunConfig
	response: str
	created_at: str

#--------------------
# Ollama call
#-------------------
OLLAMA_URL = "http://localhost:11434/api/generate"

def run_prompt(config: RunConfig) -> RunResult:
	payload = {
		"model": config.model,
		"prompt": config.prompt,
		"temperature": config.temperature,
		"stream": False,
	}
	
	with httpx.Client(timeout=60.0) as client:
		r = client.post(OLLAMA_URL, json = payload)
		r.raise_for_status()
		data = r.json()
	
	return RunResult(
		config = config,
		response = data["response"],
		created_at = datetime.now(timezone.utc).isoformat(),
	)

#------------------
# Save run
#-----------------

def save_run(result: RunResult) -> Path:
	runs_dir = Path("runs")
	runs_dir.mkdir(exist_ok=True)

	timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
	temp_str = str(result.config.temperature).replace(".", "p")
	file_path = runs_dir / f"run_{timestamp}_t{temp_str}.json"

	with open(file_path, "w") as f:
		json.dump(result.model_dump(), f, indent=2)

	return file_path

#----------------
# Main
#---------------

if __name__ == "__main__":
	console = Console()	

	prompt = "Explain what temperature does in large language models in simple terms. Give a short example."
	temps = [0.0, 0.3, 0.7, 1.1]

	console.print(f"Model: [bold]{RunConfig.model_default if hasattr(RunConfig, 'model_default') else 'llama3.1:8b'}[/bold]", style="dim")
	console.print("Running temperature comparison...\n", style="bold cyan")

	for t in temps:
		config = RunConfig(
			model = "llama3.1:8b",
			temperature = t,
			prompt = prompt,
		)
		console.print(f"[bold]Temperature:[/bold] {t}", style="yellow")
		result = run_prompt(config)
		path = save_run(result)

		console.print(
			Panel(
				result.response.strip(),
				title=f"Respone (t={t})",
				border_style="green",
			)
		)
		console.print(f"Saved run to {path}\n", style="dim")
	

	


