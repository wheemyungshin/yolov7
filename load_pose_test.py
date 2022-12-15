import json

pose_data = "/data/coco/annotations/coco_wholebody_val_v1.0.json"

with open(pose_data, 'r') as f:
    data = json.load(f)
    print(type(data))
    print(len(data))
    print(data.keys())
    print(type(data['annotations']))
    print(len(data['annotations']))
    print(type(data['annotations'][0]))
    print(data['annotations'][0])
    