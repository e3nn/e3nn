#pylint: disable=W0603,W0602
import torch
from time import perf_counter

DATA_TIMES = {}
TOTAL_TIME = 0

def clear():
    global DATA_TIMES
    global TOTAL_TIME
    DATA_TIMES = {}
    TOTAL_TIME = start()

def start():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return perf_counter()

def end(name, begin_time):
    global DATA_TIMES

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = perf_counter()
    delta = end_time - begin_time

    try:
        DATA_TIMES[name].append(delta)
    except KeyError:
        DATA_TIMES[name] = [delta]
    return end_time

def text_statistics():
    text = "[time logging] ...............unit is seconds... [tot time]/ [nbr] = [per call] [percent]\n"
    total = start() - TOTAL_TIME

    for name, times in sorted(DATA_TIMES.items()):
        text += "[time logging] {:.<30}... {: >9.3} / {: <5} = {: <10.3} {}%\n".format(
            name, sum(times),
            len(times), sum(times) / len(times),
            round(100 * sum(times) / total))
    return text
