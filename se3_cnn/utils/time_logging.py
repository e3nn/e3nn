#pylint: disable=W0603,W0602
import torch
from time import perf_counter

DATA_TIMES = {}

def clear():
    global DATA_TIMES
    DATA_TIMES = {}

def start():
    torch.cuda.synchronize()
    return perf_counter()

def end(name, begin_time):
    global DATA_TIMES

    torch.cuda.synchronize()
    end_time = perf_counter()
    delta = end_time - begin_time

    try:
        DATA_TIMES[name].append(delta)
    except KeyError:
        DATA_TIMES[name] = [delta]
    return end_time

def text_statistics():
    text = ""

    for name, times in sorted(DATA_TIMES.items()):
        text += "[time logging] {:.<30}... {: >8.2} / {: <5} = {:.2}s\n".format(name, sum(times), len(times), sum(times) / len(times))
    return text
