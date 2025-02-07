from pathlib import Path
import argparse 

import torch

import sys
sys.path.append(str(Path(__file__).parent.resolve().joinpath("../")))
from ratata.inference.video_inference import VideoInferenceYolo
from ratata.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config.yml")
    parser.add_argument("video_dir",
                        help="Path to the directory cotaining videos to inference")
    return parser.parse_args()

def main(video_path, video_inference):
    video_inference.inference_video(video_path, out_dir)

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    video_dir = args.video_dir
    config_path = args.config_path
    # load_config
    config = load_config(config_path)
    # load model
    ## オフライン処理前提なので、ymlのモデルではなく特大モデルを使用する
    model_path = Path(__file__).joinpath("../../models/yolo/yolo11x-seg.pt")
    model_path_tie = Path(__file__).joinpath("../../models/yolo/yolo11x-seg.pt")
    target_size = config.target_size
    blur = config.blur
    matt = config.matt
    enhance = config.enhance
    # video_path
    out_dir = Path(video_dir).parent.joinpath("inference")
    out_dir.mkdir(parents=True, exist_ok=True)
    for video_path in Path(video_dir).glob("*.mp4"):
        # inference
        ## インスタンスを毎回作る
        video_inference = VideoInferenceYolo(model_path, target_size, device=device,
                                            blur = blur, model_path_tie = model_path_tie,
                                            matt=matt, enhance=enhance)
        video_inference.inference_video(str(video_path), out_dir)
