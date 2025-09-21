import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def read_split_data(root: str, ratios=(0.6, 0.2, 0.2), seed=42):
    random.seed(seed)
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"
    soybean_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    soybean_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(soybean_class))
    # 保存为 JSON 文件
    with open("class_indices.json", "w") as f:
        json.dump(class_indices, f, indent=4)  # 添加此行



    # 收集所有数据路径和标签
    all_paths = []
    all_labels = []
    for cla in soybean_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in [".jpg", ".JPG", ".png", ".PNG"]]
        all_paths.extend(images)
        all_labels.extend([class_indices[cla]] * len(images))

    # 第一次划分：分出测试集
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=ratios[2],
        stratify=all_labels,
        random_state=seed
    )

    # 第二次划分：分出训练集和验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=ratios[1] / (1 - ratios[2]),  # 调整比例
        stratify=train_val_labels,
        random_state=seed
    )

    print(f"Total images: {len(all_paths)}")
    print(f"Train: {len(train_paths)} ({len(train_paths) / len(all_paths):.1%})")
    print(f"Validation: {len(val_paths)} ({len(val_paths) / len(all_paths):.1%})")
    print(f"Test: {len(test_paths)} ({len(test_paths) / len(all_paths):.1%})")

    return (
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels
    )


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
