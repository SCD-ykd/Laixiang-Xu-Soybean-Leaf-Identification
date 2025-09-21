import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import swin_tiny_patch4_window7_224 as create_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 读取类别索引
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    class_name_to_idx = {v.lower(): k for k, v in class_indict.items()}

    # 创建模型并加载权重
    model = create_model(num_classes=4).to(device)
    model_weight_path = "./weights/model-29.pth"
    weights_dict = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(weights_dict, strict=False)
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)

    print(f"Missing keys (预期内的新增模块): {missing_keys}")
    print(f"Unexpected keys (通常无害): {unexpected_keys}")###################
    model.eval()

    # 创建错误日志目录
    error_log_path = "prediction_errors.txt"
    error_img_dir = "error_images"
    os.makedirs(error_img_dir, exist_ok=True)

    with open(error_log_path, "w") as f:
        f.write("Image Path | True Label | Predicted Label | Confidence | Error Image Path\n")

    # 遍历测试图片
    img_folder_path = "D:/swin_transformer/test_images"
    assert os.path.exists(img_folder_path), f"folder: '{img_folder_path}' does not exist."

    for root, dirs, files in os.walk(img_folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                true_label_name = os.path.basename(root).lower()

                if true_label_name not in class_name_to_idx:
                    print(f"跳过未知类别: {img_path}")
                    continue

                # 读取图片并预处理
                img = Image.open(img_path).convert("RGB")
                img_tensor = data_transform(img).unsqueeze(0).to(device)

                # 预测
                with torch.no_grad():
                    output = model(img_tensor)
                    predict = torch.softmax(output, dim=1)
                    predict_cla = torch.argmax(predict).item()
                    confidence = predict[0, predict_cla].item()

                # 获取标签
                true_label = class_indict[class_name_to_idx[true_label_name]]
                predicted_label = class_indict[str(predict_cla)]

                # 显示预测结果
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.title(f"True: {true_label}\nPredicted: {predicted_label}\nConfidence: {confidence:.3f}")
                plt.axis("off")

                # 显示概率分布
                plt.subplot(1, 2, 2)
                classes = list(class_indict.values())
                probs = predict[0].cpu().numpy()
                plt.barh(classes, probs)
                plt.xlabel("Probability")
                plt.tight_layout()

                # 如果预测错误，保存图片到错误目录并记录日志
                if predicted_label != true_label:
                    error_img_path = os.path.join(error_img_dir, f"error_{os.path.basename(img_path)}")
                    img.save(error_img_path)
                    with open(error_log_path, "a") as f:
                        f.write(f"{img_path} | {true_label} | {predicted_label} | {confidence:.3f} | {error_img_path}\n")
                    plt.suptitle("⚠️ Prediction Error ⚠️", color="red")  # 在图像标题标记错误

                plt.show()
                plt.close()

if __name__ == '__main__':
    main()