from shutil import copyfile
import argparse
import os
import json
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='../data/coco/annotations', help='data root')
    parser.add_argument('--save-root', type=str, default='coco_face', help='save root')
    opt = parser.parse_args()
    
    data_root = opt.data_root
    splits = ['val', 'train']
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
        json_path = os.path.join(opt.data_root, 'coco_wholebody_{}_v1.0.json'.format(split))

        all_anno_data_dict = {}
        with open(json_path, 'r') as f:
            print("Reading:", json_path)
            json_data = json.load(f)
            data_anno = json_data['annotations']
            data_img = json_data['images']
            for anno_dict in data_anno:
                image_id = anno_dict['image_id']

                bbox = anno_dict['bbox']
                category_id = 0

                if image_id not in all_anno_data_dict:
                    temp_dict = {}
                    temp_dict['category_id'] = [category_id]
                    temp_dict['bbox'] = [bbox]
                    all_anno_data_dict[image_id] = temp_dict
                else:
                    all_anno_data_dict[image_id]['category_id'].append(category_id)
                    all_anno_data_dict[image_id]['bbox'].append(bbox)
                
                if anno_dict['face_valid']: 
                    bbox = anno_dict['face_box']
                    category_id = 1

                    if image_id not in all_anno_data_dict:
                        temp_dict = {}
                        temp_dict['category_id'] = [category_id]
                        temp_dict['bbox'] = [bbox]
                        all_anno_data_dict[image_id] = temp_dict
                    else:
                        all_anno_data_dict[image_id]['category_id'].append(category_id)
                        all_anno_data_dict[image_id]['bbox'].append(bbox)
            
            size_dict = {}
            for img_dict in data_img:
                size_dict[img_dict['id']] = (img_dict['height'], img_dict['width'])
            
            print("Writing:", opt.save_root)
            for image_id in all_anno_data_dict.keys():
                data_name = str(image_id)
                save_name = '0'*(12 - len(data_name))+data_name
                
                category_id = all_anno_data_dict[image_id]['category_id']
                bbox = all_anno_data_dict[image_id]['bbox']
                height, width = size_dict[image_id]

                labels_f = open('{}.txt'.format(os.path.join(label_dir[split], save_name)), 'w')
                #save_im = cv2.imread(os.path.join('coco/images/{}2017'.format(split), save_name+'.jpg'))
                #colors = [(255, 0, 0), (0, 0, 255)]
                for i, c_id in enumerate(category_id):
                    line = [str(c_id)]
                    line.append(str(round((bbox[i][0]+bbox[i][2]/2)/width, 6)))   
                    line.append(str(round((bbox[i][1]+bbox[i][3]/2)/height, 6)))      
                    line.append(str(round((bbox[i][2])/width, 6)))   
                    line.append(str(round((bbox[i][3])/height, 6)))   
                    labels_f.write(' '.join(line))                       
                    labels_f.write('\n')
                    #save_im = cv2.rectangle(save_im, (int(bbox[i][0]), int(bbox[i][1])), (int(bbox[i][0]+bbox[i][2]), int(bbox[i][1]+bbox[i][3])), colors[c_id], 2)
                #cv2.imwrite(os.path.join('test_vis', save_name+'.jpg'), save_im)
                labels_f.close()

                relative_path = os.path.join('./images', split, save_name+'.jpg\n')
                file_list_f[split].write(relative_path)

                copyfile(os.path.join('coco/images/{}2017'.format(split), save_name+'.jpg'), os.path.join(image_dir[split], save_name+'.jpg'))
    file_list_f[split].close()
