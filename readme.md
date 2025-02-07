# ratata  

Webカメラに繋ぎ、maskを検出し、表示するまで  
TODO: 後段のmaskを後処理する部分の実装  

## Requirements  

- Python 3.12  
- pytorch 2.5.1  

## インストール  

$ pip install torch==2.5.1 torchvision torchaudio  
$ pip install opencv_python  

# HowToUse
## main.py
$ conda activate test
$ python main.py {config_path} {cam_id}
ex. python main.py ./configs/ver2.yml 1
qを押せば終了  

### config.yml
ver1: 軽量. 多分標準PCでもうごく 
ver2: GPUノートでもギリギリって感じ

### video_inferenceについて
処理はこの1行のみ
binary_mask, result = video_inference.inference_frame(frame)
binary_maskは, ndArray[config.target_size[1], config.target_size[0]] as type(np.uint8) 値域[0 ~ 255]
で返ってくるので、frameに重畳するときは 適宜resizeして使うべし
binary_mask_resized = cv2.resize(binari_mask, (frame.shape[1], frame.shape[0]), cv2.INTER_AREA)



## 特定のvideoにinferenceをかける
$ conda activate test
$ python tools/video_inference.py {config_path} {path_to_the_video}

ex. python tools/inference_video.py ./configs/ver2.yml ./data/test_video/src/
