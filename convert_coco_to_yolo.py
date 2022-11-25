from shutil import copyfile
import argparse
import os
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/data/synthetic_cups/', help='data root')
    parser.add_argument('--save-root', type=str, default='converted_synthetic_cups', help='save root')
    parser.add_argument('--save-type', type=str, default='segments', help='txt save type: segments or boxposes')
    opt = parser.parse_args()
    
    assert opt.save_type in ['segments', 'boxposes']

    splits = ['train', 'val']
    label_dir = {}
    image_dir = {}
    file_list_f = {}
    if not os.path.exists(opt.save_root):
        os.makedirs(opt.save_root)
    for split in splits:
        label_dir[split] = os.path.join(opt.save_root, 'labels', split)
        if not os.path.exists(label_dir[split]):
            os.makedirs(label_dir[split])
        image_dir[split] = os.path.join(opt.save_root, 'images', split)
        if not os.path.exists(image_dir[split]):
            os.makedirs(image_dir[split])
        print('{}.txt'.format(os.path.join(opt.save_root, split)))
        file_list_f[split] = open('{}.txt'.format(os.path.join(opt.save_root, split)), 'w')

    data_root = opt.data_root
    json_path = os.path.join(opt.data_root, 'coco_instances.json')
    with open(json_path, 'r') as f:
        data_anno = json.load(f)['annotations']
        for data_dict in data_anno:
            segmentation = data_dict['segmentation'][0]
            image_id = data_dict['image_id']
            category_id = data_dict['category_id']
            bbox = data_dict['bbox']

            data_name = str(image_id)
            save_name = '0'*(8 - len(data_name))+data_name

            if image_id < 13500:
                split = 'train'
            else:
                split = 'val'

            labels_f = open('{}.txt'.format(os.path.join(label_dir[split], save_name)), 'w')
            if opt.save_type == 'segments':
                line = [str(category_id)]
                for seg in segmentation:    
                    line.append(str(round(seg/300, 6)))                        
                labels_f.write(' '.join(line))
                labels_f.close()

                relative_path = os.path.join('./images', split, save_name+'.jpg\n')
                file_list_f[split].write(relative_path)

            copyfile(os.path.join(data_root, 'images_raw', save_name+'.jpg'), os.path.join(image_dir[split], save_name+'.jpg'))
    file_list_f[split].close()
