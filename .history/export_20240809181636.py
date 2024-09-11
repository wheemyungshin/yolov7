import argparse
import sys
import time
import warnings

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import models
from models.experimental import attempt_load, End2End, End2End_seg
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device, intersect_dicts
from utils.add_nms import RegisterNMS
from models.yolo import Model
import collections

from models.common import Conv, DWConv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolor-csp-c.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--max-wh', type=int, default=None, help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--include-nms', action='store_true', help='export end2end onnx')
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')
    parser.add_argument('--agnostic-nms', action='store_true', help='Use Agnostic NMS')
    parser.add_argument('--seg', action='store_true', help='Segmentation-Training')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--nc', type=int, default=0, help='number of class')
    parser.add_argument('--qat', action='store_true', help='Quantization-Aware-Training')
    
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    if opt.cfg and opt.nc:
        nc = opt.nc
        ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
        if type(ckpt['model']) is Model:
            state_dict = ckpt['model'].float().state_dict()  # to FP32
        elif type(ckpt['model']) is collections.OrderedDict:
            state_dict = ckpt['model']  # to FP32
        else:
            assert (type(ckpt['model']) is Model or type(ckpt['model']) is collections.OrderedDict), "Invalid model types to load"

        model = Model(opt.cfg, ch=3, nc=nc, qat=opt.qat).to(device)  # create#, nm=nm).to(device)  # create

        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        
        for name, m in model.named_modules():
            if type(m) is Conv:
                print(name)
            '''
            if len(name.split('.')) >= 3:
                _, layer_id, layer_type = name.split('.')[:3]
                print(model.model[int(layer_id)])
            '''
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                _, layer_id = name.split('.')[:2]
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    else:
        assert not opt.cfg and not opt.nc
        model = attempt_load(opt.weights, map_location=device)  # load FP32 model
        if not hasattr(model, 'qat'):
            setattr(model, 'qat', False)
    labels = model.names

    print("Model is ", model)
    print("Model is ", type(model))

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,192,320) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    y = model(img)  # dry run
    '''
    print(y)
    print(type(y))
    print(len(y))
    print(type(y[0]))
    print(len(y[0]))
    print(type(y[1]))
    print(len(y[1]))
    print(y[0].shape)
    print(y[1][0].shape)
    print(y[1][1].shape)
    print(y[1][2].shape)
    exit()
    '''
    if opt.include_nms:
        model.model[-1].include_nms = True
        y = None

    '''
    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img, strict=False)
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    # CoreML export
    try:
        import coremltools as ct

        print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
        # convert model from torchscript and apply pixel scaling as per detect.py
        ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        bits, mode = (8, 'kmeans_lut') if opt.int8 else (16, 'linear') if opt.fp16 else (32, None)
        if bits < 32:
            if sys.platform.lower() == 'darwin':  # quantization only supported on macOS
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress numpy==1.20 float warning
                    ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
            else:
                print('quantization only supported on macOS, skipping...')

        f = opt.weights.replace('.pt', '.mlmodel')  # filename
        ct_model.save(f)
        print('CoreML export success, saved as %s' % f)
    except Exception as e:
        print('CoreML export failure: %s' % e)
                     
    # TorchScript-Lite export
    try:
        print('\nStarting TorchScript-Lite export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.ptl')  # filename
        tsl = torch.jit.trace(model, img, strict=False)
        tsl = optimize_for_mobile(tsl)
        tsl._save_for_lite_interpreter(f)
        print('TorchScript-Lite export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript-Lite export failure: %s' % e)
    '''

    # ONNX export
    import onnx

    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    if opt.grid and opt.end2end:
        option = 'full'
    elif opt.grid and not opt.end2end:
        option = 'grid_only'
    elif not opt.grid and not opt.end2end:
        option = 'no_opt'
    else:
        option = 'undefined'
    f = opt.weights.replace('.pt', '_{}_{}_{}.onnx'.format(option, opt.img_size[0], opt.img_size[1]))  # filename
    model.eval()
    output_names = ['classes', 'boxes'] if y is None else ['output']
    dynamic_axes = None
    if opt.dynamic:
        dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
            'output': {0: 'batch', 2: 'y', 3: 'x'}}
    if opt.dynamic_batch:
        opt.batch_size = 'batch'
        dynamic_axes = {
            'images': {
                0: 'batch',
            }, }
        if opt.end2end and opt.max_wh is None:
            output_axes = {
                'num_dets': {0: 'batch'},
                'det_boxes': {0: 'batch'},
                'det_scores': {0: 'batch'},
                'det_classes': {0: 'batch'},
            }
        else:
            output_axes = {
                'output': {0: 'batch'},
            }
        dynamic_axes.update(output_axes)
    elif opt.end2end and opt.max_wh is not None:
        opt.batch_size = 'batch'
        dynamic_axes = {
            'output': {0: 'batch'},
        }

    if opt.grid:
        if opt.end2end:
            print('\nStarting export end2end onnx model for %s...' % 'TensorRT' if opt.max_wh is None else 'onnxruntime')
            if opt.seg:
                model = End2End_seg(model,opt.topk_all,opt.iou_thres,opt.conf_thres,opt.max_wh,device,len(labels),opt.agnostic_nms)
            else:
                model = End2End(model,opt.topk_all,opt.iou_thres,opt.conf_thres,opt.max_wh,device,len(labels),opt.agnostic_nms)
            if opt.end2end and opt.max_wh is None:
                output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                shapes = [opt.batch_size, 1, opt.batch_size, opt.topk_all, 4,
                            opt.batch_size, opt.topk_all, opt.batch_size, opt.topk_all]
            else:
                output_names = ['output']
        else:
            model.model[-1].concat = True

    torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                        output_names=output_names,
                        dynamic_axes=dynamic_axes)

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    if opt.end2end and opt.max_wh is None:
        for i in onnx_model.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))

    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

    # # Metadata
    # d = {'stride': int(max(model.stride))}
    # for k, v in d.items():
    #     meta = onnx_model.metadata_props.add()
    #     meta.key, meta.value = k, str(v)
    # onnx.save(onnx_model, f)

    if opt.simplify:
        try:
            import onnxsim

            print('\nStarting to simplify ONNX...')
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')

    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    onnx.save(onnx_model, f)
    print('ONNX export success, saved as %s' % f)

    if opt.include_nms:
        print('Registering NMS plugin for ONNX...')
        mo = RegisterNMS(f)
        mo.register_nms()
        mo.save(f)


    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
