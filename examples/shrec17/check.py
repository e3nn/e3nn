# pylint: disable=E1101,R,C,W1202
import torch

import time
import types
import importlib.machinery


def main(model_path, batch_size):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")

    # Load the model
    loader = importlib.machinery.SourceFileLoader('model', model_path)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    model = mod.Model(55).to(device)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    model.train()
    data = torch.randn(batch_size, 1, 64, 64, 64, device=device)

    torch.cuda.synchronize()
    t = time.perf_counter()

    out = model(data)

    print(out)
    torch.cuda.synchronize()
    print("time = {:.3f}".format(time.perf_counter() - t))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    main(**args.__dict__)