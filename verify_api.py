import subprocess
import time
import hashlib
import os

def get_md5(fname):
    with open(fname, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Start server
server_proc = subprocess.Popen(
    ["/home/clemens/Dokumente/0_Master_AIR/1.WS_2025-26/Machine Learning/Satelite/.venv/bin/uvicorn", "src.dashboard.api:app", "--host", "127.0.0.1", "--port", "8005"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

time.sleep(10) # wait for startup

# Test RF
os.system("curl -s 'http://localhost:8005/api/nuremberg/experimental/heatmap/1?model=rf' --output /tmp/rf_vfinal.bin")
# Test Linear
os.system("curl -s 'http://localhost:8005/api/nuremberg/experimental/heatmap/1?model=linear' --output /tmp/linear_vfinal.bin")

server_proc.terminate()

if os.path.exists("/tmp/rf_vfinal.bin") and os.path.exists("/tmp/linear_vfinal.bin"):
    rf_hash = get_md5("/tmp/rf_vfinal.bin")
    ln_hash = get_md5("/tmp/linear_vfinal.bin")
    print(f"RF Hash: {rf_hash}")
    print(f"LN Hash: {ln_hash}")
    if rf_hash != ln_hash:
        print("SUCCESS: Hashes are different!")
    else:
        print("FAILURE: Hashes are identical!")
else:
    print("FAILURE: Binary files not created!")
