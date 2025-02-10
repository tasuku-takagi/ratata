from pathlib import Path

class SrcSelector():
    def __init__(self, bg_dir, bg_names, video_dir, video_names):
        self.bg_paths = []
        self.video_paths = []
        for bg_name in bg_names:
            bg_path = Path(bg_dir).joinpath(bg_name)
            self.bg_paths.append(str(bg_path))
        for video_name in video_names:
            video_path = Path(video_dir).joinpath(video_name)
            self.video_paths.append(str(video_path))
        self.idx = 0
        self.len_src = len(self.video_paths)

    def get_next_src(self):
        bg_path = self.bg_paths[self.idx]
        video_path = self.video_paths[self.idx]
        self.cyclic_incliment()
        print(f"src_idx: {self.idx}")
        print(f"bg: {Path(bg_path).stem}")
        print(f"video: {Path(video_path).stem}")
        return bg_path, video_path


    def cyclic_incliment(self):
        self.idx = (self.idx+1) % self.len_src
