import os
import time
import psutil
from pydantic import Field
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


class SystemMetrics(BaseModel):
	wall_time_s: float
	cpu_percent: float
	mem_rss_mb: float
	system_mem_used_mb: float
	system_mem_available_mb: float

class RunResult(BaseModel):
	config: RunConfig
	response: str
	created_at: str
	metrics: SystemMetrics

#-------------------
#Collect Metrics
#-------------------
def collect_metrics(start_time:float) -> SystemMetrics:
	proc = psutil.Process(os.getpid())

	# Percent since last call; prime it once so next call is meaningful
	proc.cpu_percent(interval=None)
	cpu_percent = proc.cpu_percent(interval=0.1)

	rss_mb = proc.memory_info().rss / (1024*1024)

	vm = psutil.virtual_memory()
	used_mb = vm.used / (1024 * 1024)
	avail_mb = vm.available / (1024 * 1024)

	return SystemMetrics(
		wall_time_s=round(time.time() - start_time, 4),
		cpu_percent=cpu_percent,
		mem_rss_mb=round(rss_mb, 2),
		system_mem_used_mb=round(used_mb, 2),
		system_mem_available_mb=round(avail_mb, 2),
	)

#--------------------
# Ollama call
#-------------------
OLLAMA_URL = "http://localhost:11434/api/generate"

def run_prompt(config: RunConfig) -> RunResult:
	start = time.time()

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
	
	metrics = collect_metrics(start)
	
	return RunResult(
		config = config,
		response = data["response"].strip(),
		created_at = datetime.now(timezone.utc).isoformat(),
		metrics = metrics,
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
	

	


