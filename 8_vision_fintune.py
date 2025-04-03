# We will be finetuning a pre-trained Mask R-CNN model on the Penn-Fudan Database for Pedestrian Detection and Segmentation.
# It contains 170 images with 345 instances of pedestrians

import os
import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate

# 这里用了 torchvision.transforms.v2，这是 PyTorch 新版的 transforms API（比旧版功能更丰富，更支持对象检测、分割等任务）。
from torchvision.transforms import v2 as T

image = read_image("data/PennFudanPed/PNGImages/FudanPed00046.png")
mask = read_image("data/PennFudanPed/PedMasks/FudanPed00046_mask.png")

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title("Image")
plt.imshow(image.permute(1, 2, 0))
plt.subplot(122)
plt.title("Mask")
plt.imshow(mask.permute(1, 2, 0))

# 自定义 PyTorch 数据集类, 用于加载 Penn-Fudan Pedestrian Detection Dataset
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))   # 读取所有图像文件名和 mask 文件名，并排序，确保图像和对应 mask 对齐
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    # 读取单个样本
    def __getitem__(self, idx):     # 加载第 idx 个图像与 mask，并构造返回的数据（图像 + 标签）
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = read_image(img_path)      # [3, H, W]
        mask = read_image(mask_path)    # [1, H, W] → encoded by color, mask 是单通道，但用颜色值表示不同目标（每个行人一个 id）

        obj_ids = torch.unqiue(mask)    # 得到不同的颜色 ID，对应不同的人
        obj_ids = obj_idx[1:]           # 去掉背景（通常为 0）
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)  # 每个 obj_id 被转换为一个 二值 mask（一个人一个 mask）。eg. 如果有两个行人，则 masks.shape == [2, H, W]

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)   # 用 torchvision 工具自动从 mask 得到 [xmin, ymin, xmax, ymax] 框

        # 构造目标标签 target
        labels = torch.ones((num_objs,), dtype=torch.int64)     # 所有目标都是行人，类编号为 1

        image_id = idx   # 图像 id
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])    # 框面积

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)   # 没有 crowd

        #  封装成 tv_tensors 类型（PyTorch 的新 API）
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # 如果提供了 transform（如随机裁剪、翻转等），则同时变换图像与 target（bounding boxes 和 masks 会一起变）
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)

# We will be using Mask R-CNN, which is based on top of Faster R-CNN. 
# Faster R-CNN is a model that predicts both bounding boxes and class scores for potential objects in the image.

# There are two common situations where one might want to modify one of the available models in TorchVision Model Zoo. 


######### 1. we want to start from a pre-trained model, and just finetune the last layer #########

# 构建一个迁移学习（transfer learning）版的 Faster R-CNN 模型，并用于你自己的目标检测任务，比如 Penn-Fudan 数据集中只检测“行人”

# 加载一个 预训练的 Faster R-CNN 模型（在 COCO 上训练过），backbone 是 ResNet50 + FPN，weights="DEFAULT" 表示使用官方 COCO 数据集上训练好的权重
# 这个模型的输出头（box predictor）默认是 检测 COCO 的 91 个类
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT") 

# 替换分类头（适配我们自己的任务）
num_classes = 2     # 1 class (person) + background

# 获取当前分类器的输入维度
in_features = model.roi_heads.box_predictor.cls_score.in_features   # cls_score 是最后一层全连接层（用于分类），我们需要知道它的输入大小（in_features），以便构建新的一层

# 替换分类器头，输入维度和原来保持一致（保持 backbone 输出兼容），输出类别数换成我们自己的（这里是 2）
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


#### 2 - Modifying the model to add a different backbone ####
# 略

# Object detection and instance segmentation model for PennFudan Dataset
# In our case, we want to finetune from a pre-trained model, given that our dataset is very small, so we will be following approach number 1.

# 用于构建一个 可迁移学习的 Mask R-CNN 模型，适用于实例分割任务（不仅检测每个物体的位置，还要分割出它的轮廓）。
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    # box predictor（用于边界框分类 + 回归）, mask predictor（用于分割掩码）
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")   # 加载 Mask R-CNN + ResNet-50 + FPN 的实例分割模型

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

# 接收一个布尔参数 train：是否处于训练阶段。
# 训练时我们会加上数据增强（比如随机翻转）, 测试时则只做标准的预处理，不加入增强操作
def get_transform(train):
    transforms = []

    if train:   #  随机水平翻转（仅用于训练）
        transforms.append(T.RandomHorizontalFlip(0.5))  # 以 50% 概率 将图像水平翻转，并自动同步调整 bounding boxes、masks、labels 等 target。

    # 把图像从整数（uint8）变成浮点数（float32）, scale=True 会自动将像素值从 [0, 255] 缩放到 [0.0, 1.0] 区间
    transforms.append(T.ToDtype(torch.float, scale=True))

    # 将图像包装为 纯 tensor 类型，并剥离掉 tv_tensors.Image 的额外属性（如 size、format 等
    transforms.append(T.ToPureTensor())

    # 返回一个可执行的 组合变换对象
    return T.Compose(transforms)

# 完整的训练流程
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2

# use our dataset and defined transformations
dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('data/PennFudanPed', get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()                 # 打乱索引
dataset = torch.utils.data.Subset(dataset, indices[:-50])       # 拿前面所有数据作为训练集，最后 50 个样本作为测试集
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,                            # 一次训练两个样本
    batch_size=2,                       
    shuffle=True,                       # 打乱数据
    collate_fn=utils.collate_fn         # 用于合并不同大小的目标检测样本
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

# 使用前面定义的函数，返回一个基于 COCO 预训练权重的 Mask R-CNN，并替换了分类头和 mask 头
model = get_model_instance_segmentation()

model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]     # 只优化需要梯度更新的参数
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,   
    weight_decay=0.0005     # 权重衰减
)

# lr scheduler, 每训练 3 轮（step_size=3）将学习率 ×0.1
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

num_epochs = 2

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

print("That's it!")