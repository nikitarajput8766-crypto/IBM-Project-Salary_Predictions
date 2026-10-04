import subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
result = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)
