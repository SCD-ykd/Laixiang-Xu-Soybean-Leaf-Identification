import torch
from model import swin_tiny_patch4_window7_224 as create_model

def count_parameters(model, verbose=True):
    """统计PyTorch模型的参数量（总参数量/可训练参数量/非可训练参数量）"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    if verbose:
        # 打印各层参数量明细
        print("\n" + "=" * 60)
        print(f"{'Layer Name':<30} | {'Parameters':>15} | {'Trainable':>10}")
        print("-" * 60)
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.LayerNorm)):
                params = sum(p.numel() for p in module.parameters())
                trainable = any(p.requires_grad for p in module.parameters())
                print(f"{name[:28]:<30} | {params:>15,} | {'Yes' if trainable else 'No':>10}")
        print("=" * 60)

        # 打印汇总信息
        print(f"\nTotal Parameters       : {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable Parameters   : {trainable_params:,}")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")

    return total_params

# 创建模型并测量参数量
model = create_model(num_classes=4)  # 替换为你的类别数
total_params = count_parameters(model)

# 自动转换单位
def format_params(num_params):
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    else:
        return f"{num_params/1e3:.1f}K"

print(f"\nModel Size: {format_params(total_params)}")