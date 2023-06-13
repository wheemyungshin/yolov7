import argparse
import torch.fx
import torch.nn as nn
import models_non_dynamic_flow
from models_non_dynamic_flow.experimental import attempt_load
from utils.torch_utils import TracedModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', nargs='+', type=int, default=[192,256], help='inference size (pixels)')
    parser.add_argument('--save', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    opt = parser.parse_args()
    print(opt)

    model = attempt_load(weights=opt.weights)
    #graph = TracedModel(model, 'cpu', tuple([opt.img_size[0], opt.img_size[1]]))

    img = torch.zeros(1, 3, opt.img_size[0], opt.img_size[1])  # image size(1,3,192,320) iDetection
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models_non_dynamic_flow.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    y = model(img, augment=False)
    model.eval()
    #torch.fx.Tracer.is_leaf_module(model)
    #graph = torch.jit.trace(model, img, strict=False)
    
    graph = torch.fx.Tracer().trace(model)
    traced_model = torch.fx.GraphModule(model, graph)

    torch.save(traced_model, opt.save)