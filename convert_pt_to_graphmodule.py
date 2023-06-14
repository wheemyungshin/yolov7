import argparse
import torch.fx
import torch.nn as nn
import models_non_dynamic_flow
from models_non_dynamic_flow.yolo import Model
from utils.torch_utils import TracedModel
import yaml


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--save', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    opt = parser.parse_args()

    nc = 8

    trained_model = torch.load(opt.weights)['model'].float().train()  # load checkpoint
    model = Model(opt.cfg, ch=3, nc=nc)  # create
    #exclude = ['anchor']
    #state_dict = ckpt['model'].float().state_dict()  # to FP32
    #state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(trained_model.state_dict(), strict=False)  # load

    graph = torch.fx.Tracer().trace(model)
    traced_model = torch.fx.GraphModule(model, graph)

    print(type(traced_model))

    torch.save(traced_model, opt.save)
    print("saved: ", opt.save)