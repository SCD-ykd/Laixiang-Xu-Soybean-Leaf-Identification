import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model import swin_tiny_patch4_window7_224 as create_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载类别映射
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"文件 '{json_path}' 不存在。"
    with open(json_path, "r", encoding="utf-8") as f:
        class_indict = json.load(f)
    inverted_class_indict = {v: k for k, v in class_indict.items()}

    # 创建模型并加载权重
    model = create_model(num_classes=len(class_indict)).to(device)
    model_weight_path = "./weights/model-49.pth"
    weights_dict = torch.load(model_weight_path, map_location=device, weights_only=True)
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)

    print(f"Missing keys (预期内的新增模块): {missing_keys}")
    print(f"Unexpected keys (通常无害): {unexpected_keys}")
    model.eval()

    # 数据预处理
    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 读取测试集
    test_set_path = "./test_set.txt"
    assert os.path.exists(test_set_path), "测试集文件不存在。"
    test_images, test_labels = [], []
    with open(test_set_path, "r", encoding="utf-8") as f:
        next(f)  # 跳过标题行
        for line in f:
            path, label = line.strip().split("|")
            test_images.append(path)
            test_labels.append(int(label))

    # 初始化统计字典和结果文件
    class_stats = {cls: {"total": 0, "correct": 0, "errors": []} for cls in class_indict.keys()}
    result_file = "predict_results.txt"

    # 写入结果文件表头
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("图像路径|真实标签|预测标签|预测概率|是否正确\n")

    # 批量预测
    total = len(test_images)
    correct = 0
    for img_path, true_label in zip(test_images, test_labels):
        true_label_str = str(true_label)
        if not os.path.exists(img_path):
            print(f"警告：图片 '{img_path}' 不存在，已跳过。")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = data_transform(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"处理图片 '{img_path}' 时出错: {str(e)}")
            continue

        with torch.no_grad():
            output = torch.squeeze(model(img_tensor))
            probabilities = torch.softmax(output, dim=0)  # 获取每个类别的概率
            pred_class = torch.argmax(output).item()
            pred_label_str = str(pred_class)
            pred_prob = probabilities[pred_class].item()  # 获取预测类别的概率
        is_correct = pred_class == true_label  # 定义 is_correct

        # 更新统计信息
        true_label_str = inverted_class_indict[true_label]
        if is_correct:
            correct += 1
            class_stats[true_label_str]["correct"] += 1
        else:
            class_stats[true_label_str]["errors"].append(pred_label_str)  # 记录错误预测的类别
        class_stats[true_label_str]["total"] += 1

        # 写入单条结果
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(f"{img_path}|{true_label}|{pred_class}|{pred_prob:.4f}|{is_correct}\n")

    # 计算并写入统计信息
    with open(result_file, "a", encoding="utf-8") as f:
        f.write("\n==== 类别统计 ====\n")
        for cls_idx, stats in class_stats.items():
            if stats["total"] == 0:
                continue
            accuracy = stats["correct"] / stats["total"]
            cls_name = class_indict[cls_idx]
            f.write(
                f"类别 {cls_name} ({cls_idx}): "
                f"正确数 {stats['correct']}/{stats['total']} "
                f"正确率 {accuracy:.4f}\n"
            )

        # 总体统计
        overall_accuracy = correct / total if total > 0 else 0
        f.write(f"\n总体正确率: {overall_accuracy:.4f} ({correct}/{total})\n")

        # 错误预测统计
        f.write("\n==== 错误预测统计 ====\n")
        for cls_idx, stats in class_stats.items():
            if stats["total"] == 0:
                continue
            cls_name = class_indict[cls_idx]
            if stats["errors"]:
                error_counts = {}
                for error in stats["errors"]:
                    error_key = int(error)
                    error_cls_name = inverted_class_indict.get(error_key, "Unknown")
                    error_counts[error_cls_name] = error_counts.get(error_cls_name, 0) + 1
                f.write(f"类别 {cls_name} ({cls_idx}) 错误预测为：\n")
                for error_cls_name, count in error_counts.items():
                    f.write(f"  类别 {error_cls_name}: {count} 次\n")
            else:
                f.write(f"类别 {cls_name} ({cls_idx}) 没有错误预测。\n")

    print(f"预测结果已保存至 {result_file}")


if __name__ == '__main__':
    main()