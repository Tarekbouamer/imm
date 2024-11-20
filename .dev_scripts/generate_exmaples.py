import subprocess
from pathlib import Path

from imm.settings import MATCHERS_LIST


def run_command(cmd: list, params: dict):
    """run command with parameters."""
    for key, value in params.items():
        cmd.append(f"--{key}")
        if value is not None:
            cmd.append(str(value))
    try:
        subprocess.run(cmd, check=True)
    except:
        raise


for emii in MATCHERS_LIST:
    extractor = emii.extractor
    matcher = emii.matcher
    img0_path = emii.img0
    img1_path = emii.img1

    save_path = Path("assets/examples")
    save_path.mkdir(exist_ok=True, parents=True)

    if extractor is None:
        save_path = save_path / f"{matcher}.png"
    else:
        save_path = save_path / f"{extractor}_{matcher}.png"

    # Parameters
    params = {
        "matcher": matcher,
        "max_keypoints": 1000,
        "max_img_size": 640,
        "save_path": str(save_path),
    }

    if extractor is not None:
        params["extractor"] = extractor

    cmd = ["python", "-m", "imm.tools.match", img0_path, img1_path]

    run_command(cmd, params)
