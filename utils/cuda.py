import os

import torch

# 环境原因
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for i in range(torch.cuda.device_count()):
    device = torch.device("cuda:" + str(i) if torch.cuda.is_available() else "cpu")
    print("GPU 编号: {}".format(device))
    print("GPU 名称: {}".format(torch.cuda.get_device_name(i)))
    print("GPU total 容量: {} G".format(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024))
    torch.tensor(0.0).to(device)