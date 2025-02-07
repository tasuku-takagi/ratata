from pathlib import Path
import argparse 


import sys
sys.path.append(str(Path(__file__).parent.resolve().joinpath("../")))
from ratata.inference.video_inference import VideoInferenceYolo


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze profile data")
    parser.add_argument("video_dir",
                        help="Path to the video to inference")
    return parser.parse_args()

def main(video_path, video_inference):
    video_inference.inference_video(video_path, out_dir)

if __name__ == "__main__":
    args = parse_args()
    video_dir = args.video_dir
    # load model
    model_path = Path(__file__).joinpath("../../models/yolo/yolo11x-seg.pt")
    model_path_tie = Path(__file__).joinpath("../../models/yolo/yolo11x-seg.pt")
    target_size = (1280, 720)
    device = "cuda:0"
    blur = ""
    matt = True
    # matt=False
    enhance = True
    video_inference = VideoInferenceYolo(model_path, target_size, device=device,
                                         blur = blur, model_path_tie = model_path_tie,
                                         matt=matt, enhance=enhance)
    # video_path
    out_dir = Path(video_dir).parent.joinpath("inference")
    out_dir.mkdir(parents=True, exist_ok=True)
    for video_path in Path(video_dir).glob("*.mp4"):
        # inference
        video_inference.inference_video(str(video_path), out_dir)
