#pylint: disable=E,W
from pycuda import autoinit
from pycuda.driver import mem_get_info
import torch

DATA_MEM = mem_get_info()

def format_memory(n):
    sign = 1 if n >= 0 else -1
    n = n if n >= 0 else -n
    n = float(n)
    suffix = ["", "kiB", "MiB", "GiB"]
    for i in range(4):
        if n < 1000:
            return "{:.4}{}".format(sign * n, suffix[i])
        n /= 1024
    return "{:.4}TiB".format(sign * n)

def used_memory():
    mem = mem_get_info()
    return mem[1] - mem[0]

def measure_gpu_memory(ident=""):
    global DATA_MEM
    torch.cuda.synchronize()
    old = DATA_MEM[1] - DATA_MEM[0]
    mem = mem_get_info()
    now = mem[1] - mem[0]
    text = "[Memory] {} {} + {} = {}".format(ident, format_memory(old), format_memory(now - old), format_memory(now))
    DATA_MEM = mem
    return text
