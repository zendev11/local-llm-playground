import os
import time
import psutil
import json
import re
import subprocess
import httpx
from datetime import datetime, timezone
from pydantic import Field
from pathlib import Path
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

	swap_used_mb:float
	swap_total_mb: float
	memory_pressure: str # "low" | "medium" | "high"
	vm_available_mb: float
	vm_compressed_mb: float

class RunResult(BaseModel):
	config: RunConfig
	response: str
	created_at: str
	metrics: SystemMetrics



#-------------------
#Collect Metrics
#-------------------
def _read_vm_stat() -> dict:
	"""
	Parse macOS vm_stat output and return:
	- page_size_bytes
	- pages_free
	- pages_speculative
	- pages_compressed
	"""
	out = subprocess.check_output(["vm_stat"], text=True)

	# First line looks like: "match Virtual Memory Statistics: (page size of 16384 bytes)"
	m = re.search(r"page size of (\d+) bytes", out)
	page_size = int(m.group(1)) if m else 4096

	def get_pages(label: str) -> int:
		mm = re.search(rf"{re.escape(label)}:\s+(\d+)\.", out)
		return int(mm.group(1)) if mm else 0
	
	pages_free = get_pages("Pages free")
	pages_spec = get_pages("Pages speculative")
	pages_comp = get_pages("Pages occupied by compressor")

	return {
		"page_size": page_size,
		"pages_free": pages_free,
		"pages_spec": pages_spec,
		"pages_comp": pages_comp
	}

def _pressure_level(system_available_mb: float, swap_used_mb: float) -> str:
	if swap_used_mb > 256 or system_available_mb < 1024:
		return "high"
	if swap_used_mb > 0 or system_available_mb < 2048:
		return "medium"
	return "low"
	
def collect_metrics(start_time:float) -> SystemMetrics:
	proc = psutil.Process(os.getpid())

	# Percent since last call; prime it once so next call is meaningful
	proc.cpu_percent(interval=None)
	cpu_percent = proc.cpu_percent(interval=0.1)

	rss_mb = proc.memory_info().rss / (1024*1024)

	vm = psutil.virtual_memory()
	used_mb = vm.used / (1024 * 1024)
	avail_mb = vm.available / (1024 * 1024)

	sm = psutil.swap_memory()
	swap_used_mb = sm.used / (1024 * 1024)
	swap_total_mb = sm.total / (1024 * 1024)

	# macOS-specific memory pressure-ish signals
	vmstat = _read_vm_stat()
	page_size = vmstat["page_size"]
	vm_available_bytes = (vmstat["pages_free"]+ vmstat["pages_spec"]) * page_size
	vm_compressed_bytes = vmstat["pages_comp"] * page_size

	vm_available_mb = vm_available_bytes / (1024 * 1024)
	vm_compressed_mb = vm_compressed_bytes / (1024 * 1024)

	pressure = _pressure_level(avail_mb, swap_used_mb)

	return SystemMetrics(
		wall_time_s=round(time.time() - start_time, 4),
		cpu_percent=cpu_percent,
		mem_rss_mb=round(rss_mb, 2),
		system_mem_used_mb=round(used_mb, 2),
		system_mem_available_mb=round(avail_mb, 2),
		swap_used_mb=round(swap_used_mb, 2),
		swap_total_mb=round(swap_total_mb, 2),
		memory_pressure=pressure,
		vm_available_mb=round(vm_available_mb,2),
		vm_compressed_mb=round(vm_compressed_mb, 2),
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
	

	


