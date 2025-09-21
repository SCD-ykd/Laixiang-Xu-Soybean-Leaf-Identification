import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from model import swin_tiny_patch4_window7_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate

def convert_to_rgb(img):
    return img.convert('RGB')
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log_file = "training_log.txt"
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
        # 初始化日志文件，清空内容

        with open(log_file, 'w') as f:
            f.write("")  # 清空或创建新文件

    tb_writer = SummaryWriter()
    # ==== 唯一的数据划分调用 ====
    train_images_path, train_images_label, \
        val_images_path, val_images_label, \
        test_images_path, test_images_label = read_split_data(
        root=args.data_path,
        ratios=(0.6, 0.2, 0.2)
    )

    # 保存测试集
    with open("test_set.txt", "w", encoding="utf-8") as f:
        f.write("path|label\n")
        for path, label in zip(test_images_path, test_images_label):
            f.write(f"{path}|{label}\n")

    img_size = 224
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),  # 与验证集一致，缩放到256
            transforms.CenterCrop(img_size),  # 中心裁剪到224x224
            transforms.Lambda(convert_to_rgb),  # 保留RGB转换
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.Lambda(convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)
    # 初始化分类头（关键！）
    if hasattr(model, "head") and isinstance(model.head, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(model.head.weight, mode="fan_out")
        if model.head.bias is not None:
            torch.nn.init.constant_(model.head.bias, 0)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        # 写入日志文件
        log_line = f"epoch:{epoch},train_loss:{train_loss:.4f},train_acc:{train_acc:.4f},val_loss:{val_loss:.4f},val_acc:{val_acc:.4f}\n"
        with open(log_file, 'a') as f:
            f.write(log_line)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="D:\swin transformer3\soybean")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
