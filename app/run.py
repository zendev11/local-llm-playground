from datetime import datetime
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
		created_at = datetime.utcnow().isoformat(),
	)

#------------------
# Save run
#-----------------

def save_run(result: RunResult) -> Path:
	runs_dir = Path("runs")
	runs_dir.mkdir(exist_ok=True)

	timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
	file_path = runs_dir / f"run_{timestamp}.json"

	with open(file_path, "w") as f:
		json.dump(result.model_dump(), f, indent=2)

	return file_path

#----------------
# Main
#---------------

if __name__ == "__main__":
	console = Console()

	config = RunConfig(
		prompt = "Say 'Ollama is running' and nothing else."
	)

	console.print("Running prompt...", style="bold cyan")

	result = run_prompt(config)
	path = save_run(result)

	console.print(
		Panel(
		   result.response.strip(),
		   title="Model Response",
		   border_style = "green",
		)
	)

	console.print(f"Saved run to {path}", style="dim")

