import os
import numpy as np
import json

root = "eval_output/waymo_nvs"
scenes = ["0017085", "0145050", "0147030", "0158150"]

eval_dict = {
    "TEST": {"psnr": [], "ssim": [], "lpips": []},
}
for scene in scenes:
    eval_dir = os.path.join(root, scene, "eval")
    dirs = os.listdir(eval_dir)
    test_path = sorted([d for d in dirs if d.startswith("test")], key=lambda x: int(x.split("_")[1]))[-1]
    for name, path in [("TEST", test_path)]:
        psnrs = []
        ssims = []
        lpipss = []
        with open(os.path.join(eval_dir, path, "metrics.json"), "r") as f:
            data = json.load(f)
        eval_dict[name]["psnr"].append(data["psnr"])
        eval_dict[name]["ssim"].append(data["ssim"])
        eval_dict[name]["lpips"].append(data["lpips"])
        
print(f'TEST PSNR:{np.mean(eval_dict["TEST"]["psnr"]):.3f} SSIM:{np.mean(eval_dict["TEST"]["ssim"]):.3f} LPIPS:{np.mean(eval_dict["TEST"]["lpips"]):.3f}')
