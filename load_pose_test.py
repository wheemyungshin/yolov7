import json

pose_data = "/coco/annotations/person_keypoints_val2017.json"

with open(pose_data, 'r') as f:
    data = json.load(f)
    print(type(data))
    print(len(data))
    print(data.keys())
    print(type(data['annotations']))
    print(len(data['annotations']))
    print(type(data['annotations'][0]))
    print(data['annotations'][0])
    for annos in data['annotations']:
        for i in range(2,51,3):
            if annos['keypoints'][i] != 0:
                print(i)
                print(annos['keypoints'][i-2], annos['keypoints'][i-1])
    