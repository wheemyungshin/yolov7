from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import select_device, is_parallel
import yaml
from models.experimental import attempt_load, fuse_modules

device = select_device('0', batch_size=1)
# model trained by cfg/training/*.yaml
#ckpt = torch.load('weights/cityscapes_ADAS_only_with_bicyclebustruck_updated_easy_IDDv3_vehicle_only_test_india_yolov7-mobilenet_nofuse_easyaugment_smallest_min_s384_384_3vcls_qat_e123.pt', map_location=device)
#ckpt = torch.load('weights/yolov7-mobilenet.pt', map_location=device)
ckpt = attempt_load('weights/yolov7-mobilenet.pt', map_location=device)
# reparameterized model in cfg/deploy/*.yaml
model = Model('cfg/training/yolov7-mobilenet.yaml', ch=3, nc=80).to(device)

with open('cfg/training/yolov7-mobilenet.yaml') as f:
    yml = yaml.load(f, Loader=yaml.SafeLoader)
anchors = len(yml['anchors'][0]) // 2

print(ckpt.keys())

# copy intersect weights
state_dict = ckpt['model'].float().state_dict()
exclude = []
intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
model.load_state_dict(intersect_state_dict, strict=False)
#model.names = ckpt['model'].names
#model.nc = ckpt['model'].nc

print("BEF: ", state_dict.keys())
'''
# reparametrized YOLOR
for i in range((model.nc+5)*anchors):
    model.state_dict()['model.105.m.0.weight'].data[i, :, :, :] *= state_dict['model.105.im.0.implicit'].data[:, i, : :].squeeze()
    model.state_dict()['model.105.m.1.weight'].data[i, :, :, :] *= state_dict['model.105.im.1.implicit'].data[:, i, : :].squeeze()
    model.state_dict()['model.105.m.2.weight'].data[i, :, :, :] *= state_dict['model.105.im.2.implicit'].data[:, i, : :].squeeze()
model.state_dict()['model.105.m.0.bias'].data += state_dict['model.105.m.0.weight'].mul(state_dict['model.105.ia.0.implicit']).sum(1).squeeze()
model.state_dict()['model.105.m.1.bias'].data += state_dict['model.105.m.1.weight'].mul(state_dict['model.105.ia.1.implicit']).sum(1).squeeze()
model.state_dict()['model.105.m.2.bias'].data += state_dict['model.105.m.2.weight'].mul(state_dict['model.105.ia.2.implicit']).sum(1).squeeze()
model.state_dict()['model.105.m.0.bias'].data *= state_dict['model.105.im.0.implicit'].data.squeeze()
model.state_dict()['model.105.m.1.bias'].data *= state_dict['model.105.im.1.implicit'].data.squeeze()
model.state_dict()['model.105.m.2.bias'].data *= state_dict['model.105.im.2.implicit'].data.squeeze()

print("AFT: ", model)

# model to be saved
ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
        'optimizer': None,
        'training_results': None,
        'epoch': -1}

# save reparameterized model
torch.save(ckpt, 'weights/yolov7_reparam.pt')
'''