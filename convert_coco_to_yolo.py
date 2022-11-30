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
    all_anno_data_dict = {}
    with open(json_path, 'r') as f:
        data_anno = json.load(f)['annotations']
        for anno_dict in data_anno:
            image_id = anno_dict['image_id']
            segmentation = anno_dict['segmentation'][0]
            category_id = anno_dict['category_id']
            bbox = anno_dict['bbox']

            if image_id not in all_anno_data_dict:
                temp_dict = {}
                temp_dict['segmentation'] = [segmentation]
                temp_dict['category_id'] = [category_id]
                temp_dict['bbox'] = [bbox]
                all_anno_data_dict[image_id] = temp_dict
            else:
                all_anno_data_dict[image_id]['segmentation'].append(segmentation)
                all_anno_data_dict[image_id]['category_id'].append(category_id)
                all_anno_data_dict[image_id]['bbox'].append(bbox)

            
    for image_id in all_anno_data_dict.keys():
        data_name = str(image_id)
        save_name = '0'*(8 - len(data_name))+data_name

        if image_id < 13500:
            split = 'train'
        else:
            split = 'val'
        
        segmentation = all_anno_data_dict[image_id]['segmentation']
        category_id = all_anno_data_dict[image_id]['category_id']
        bbox = all_anno_data_dict[image_id]['bbox']

        labels_f = open('{}.txt'.format(os.path.join(label_dir[split], save_name)), 'w')
        for i, c_id in enumerate(category_id):
            if opt.save_type == 'segments':
                line = [str(c_id)]
                for seg in segmentation[i]:    
                    line.append(str(round(seg/300, 6)))                        
                labels_f.write(' '.join(line))                       
                labels_f.write('\n')
        labels_f.close()

        relative_path = os.path.join('./images', split, save_name+'.jpg\n')
        file_list_f[split].write(relative_path)

        copyfile(os.path.join(data_root, 'images_raw', save_name+'.jpg'), os.path.join(image_dir[split], save_name+'.jpg'))
    file_list_f[split].close()
