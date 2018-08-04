# -*- coding=utf-8 -*-
import json
import os
import cv2
import xml.etree.ElementTree as ET
import shutil

# 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(xml_path):
    tree = ET.parse(xml_path)		
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(box[0].text)
        y_min = int(box[1].text)
        x_max = int(box[2].text)
        y_max = int(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords

def convert(root_path, source_xml_root_path, target_xml_root_path, phase='train', split=80000):
    '''
    root_path:
        根路径，里面包含JPEGImages(图片文件夹)，classes.txt(类别标签),以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
    source_xml_root_path:
        VOC xml文件存放的根目录
    target_xml_root_path:
        coco xml存放的根目录
    phase:
        状态：'train'或者'test'
    split:
        train和test图片的分界点数目

    '''

    dataset = {'categories':[], 'images':[], 'annotations':[]}

    # 打开类别标签
    with open(os.path.join(root_path, 'classes.txt')) as f:
        classes = f.read().strip().split()

    # 建立类别标签和数字id的对应关系
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'beverage'})   #mark

    # 读取images文件夹的图片名称
    pics = [f for f in os.listdir(os.path.join(root_path, 'JPEGImages'))]

    # 判断是建立训练集还是验证集
    if phase == 'train':
        pics = [line for i, line in enumerate(pics) if i <= split]
    elif phase == 'val':
        pics = [line for i, line in enumerate(pics) if i > split]

    print('---------------- start convert ---------------')
    bnd_id = 1	#初始为1
    for i, pic in enumerate(pics):
        # print('pic  '+str(i+1)+'/'+str(len(pics)))
        xml_path = os.path.join(source_xml_root_path, pic[:-4]+'.xml')
        pic_path = os.path.join(root_path, 'JPEGImages/' + pic)
        # 用opencv读取图片，得到图像的宽和高
        im = cv2.imread(pic_path)
        height, width, _ = im.shape
        # 添加图像的信息到dataset中
        dataset['images'].append({'file_name': pic,
                                  'id': i,
                                  'width': width,
                                  'height': height})
        try:
            coords = parse_xml(xml_path)
        except:
            print(pic[:-4]+'.xml not exists~')
            continue
        for coord in coords:
            # x_min
            x1 = int(coord[0])-1
            x1 = max(x1, 0)
            # y_min
            y1 = int(coord[1])-1
            y1 = max(y1, 0)
            # x_max
            x2 = int(coord[2])
            # y_max
            y2 = int(coord[3])
            assert x1<x2
            assert y1<y2
            # name
            name = coord[4]
            cls_id = classes.index(name)+1	#从1开始
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            dataset['annotations'].append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': int(cls_id),
                'id': bnd_id,
                'image_id': i,
                'iscrowd': 0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })
            bnd_id += 1

    # 保存结果的文件夹
    folder = os.path.join(target_xml_root_path, 'annotations')
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    json_name = os.path.join(target_xml_root_path, 'annotations/instances_{}2014.json'.format(phase))
    with open(json_name, 'w') as f:
      json.dump(dataset, f)

if __name__ == '__main__':
    convert(root_path='./data_voc/VOC2007', source_xml_root_path = './data_voc/VOC2007/Annotations', target_xml_root_path = './data_coco')