import torch
from model import swin_tiny_patch4_window7_224 as create_model
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time


def main():
    # 设备配置（自动选择GPU或CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化模型（假设分类任务有4类）
    model = create_model(num_classes=4).to(device)
    model.eval()  # 切换到评估模式

    # 定义输入张量（batch_size=1, 通道=3, 高=224, 宽=224）
    input_size = (1, 3, 224, 224)
    dummy_input = torch.randn(input_size).to(device)

    # ==============================
    # 1. 计算参数量（Params）
    # ==============================
    print("\n" + "=" * 50)
    print("模型参数量统计：")
    summary(model, input_size=input_size, device=device, col_names=["input_size", "output_size", "num_params"])

    # ==============================
    # 2. 计算FLOPs（浮点运算次数）
    # ==============================
    print("\n" + "=" * 50)
    print("FLOPs分析：")
    flops = FlopCountAnalysis(model, dummy_input)
    print(flop_count_table(flops))  # 打印分层FLOPs统计表
    print(f"总FLOPs: {flops.total() / 1e9:.2f} G")  # 转换为十亿单位（GigaFLOPs）

    # ==============================
    # 3. 测量推理速度（单位：秒/样本）
    # ==============================
    print("\n" + "=" * 50)
    print("测量推理速度...")

    # 预热（避免冷启动误差）
    for _ in range(10):
        _ = model(dummy_input)

    # 正式计时（100次推理取平均）
    total_time = 0
    runs = 100
    for _ in range(runs):
        start_time = time.time()
        _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 确保CUDA操作同步
        total_time += (time.time() - start_time)

    avg_time = total_time / runs
    print(f"平均推理时间: {avg_time:.4f} 秒/样本")
    print(f"推理速度: {1 / avg_time:.2f} 样本/秒")


if __name__ == '__main__':
    main()