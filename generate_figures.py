import subprocess
import sys

scripts = [
    "plots/plots_mccall.py",
    "plots/plots_lq.py",
    "plots/plots_rbc.py",
]

for script in scripts:
    print(f"Running {script} ...")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"ERROR: {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"Done: {script}")

print("All figures generated.")
