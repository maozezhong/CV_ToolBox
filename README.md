## 文件
- DataAugForObjectDetection : 针对目标检测的增强脚本文件夹，增强方式包括
    - 裁剪（会改变bbox）
    - 平移（会改变bbox）
    - 旋转（会改变bbox）
    - 镜像（会改变bbox）
    - 改变亮度
    - 加噪声
    - [cutout](https://arxiv.org/abs/1708.04552)
- KITTI_2_VOC : 将KITTI数据形式转换为VOC形式
- VOC_2_COCO : 讲VOC形式数据转换为COCO格式
- Attention : feature map Attention

## To_Do_List
- [ ] GAN with LSR for data augment
