"""
server.py — CLI entry point. Wraps server_core.ServerManager.
Usage:
    python server.py [--cpu] [--quant] [--index -1] [--port 4396]
"""

import argparse
import time

from server_core import ServerManager

parser = argparse.ArgumentParser(description="rotate-captcha-crack inference server")
parser.add_argument("--index", "-i", type=int, default=-1, help="Model index (-1 = latest)")
parser.add_argument("--quant", action="store_true", help="Use INT8 quantized model (CPU)")
parser.add_argument("--cpu",   action="store_true", help="Force FP32 model on CPU (no GPU required)")
parser.add_argument("--port",  "-p", type=int, default=4396, help="HTTP port (default 4396)")
opts = parser.parse_args()

if opts.quant:
    mode = "quant"
elif opts.cpu:
    mode = "cpu"
else:
    mode = "gpu"

mgr = ServerManager()
mgr.set_log_callback(print)
mgr.start(mode=mode, port=opts.port, index=opts.index)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    mgr.stop()
