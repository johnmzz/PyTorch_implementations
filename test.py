import torch
import torch_geometric

# 查看 PyTorch 版本
print("PyTorch version:", torch.__version__)

# 查看是否支持 CUDA
print("CUDA available:", torch.cuda.is_available())

# 查看 CUDA 版本（PyTorch 编译用的版本）
print("CUDA version (compile time):", torch.version.cuda)

# 查看当前设备的 GPU 名称
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))
    print("CUDA device count:", torch.cuda.device_count())

# 查看 PyG 版本
print("PyG version:", torch_geometric.__version__)
