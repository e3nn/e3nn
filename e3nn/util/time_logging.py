# pylint: disable=no-member, missing-docstring, invalid-name, redefined-builtin, arguments-differ, line-too-long, unbalanced-tuple-unpacking
from time import perf_counter
import torch

DATA_TIMES = {}


def clear():
    global DATA_TIMES
    DATA_TIMES = {}


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
    total = max(sum(times) for _, times in DATA_TIMES.items())

    for name, times in sorted(DATA_TIMES.items(), key=lambda x: sum(x[1]), reverse=True):
        text += "[time logging] {:.<30}... {: >9.3} / {: <5} = {: <10.3} {}%\n".format(
            name, sum(times),
            len(times), sum(times) / len(times),
            round(100 * sum(times) / total))
    return text
