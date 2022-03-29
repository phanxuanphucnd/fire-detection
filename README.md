# Yolov5 TensorRT

YOLOv5 conversion and inference using TensorRT (FP16), with no complicated installations setup and zero precession loss!

## Inference with TensorRT

```bash
$ pip install -r requirements.txt
```

Now run inference on video or image file (with pretrained weights).

```bash
python detect.py --input $PATH_TO_INPUT_FILE --output $OUTPUT_FILE_NAME
```

<br>

You can also pass ```--weights``` to use your own custom onnx weight file (it'll generate tensorrt engine file internally) or tensorrt engine file (generated from convert.py). You can also pass ```--classes``` for your custom trained weights and/or to filter classes for COCO.

For pretrained default weights (```--weights yolov5s```), scripts will download + internally generate new engine file for unseen input shape, but if you are using a custom weight then remeber to rename or remove engine file if you want to generate engines for different shapes. 

## Convert ONNX to TensorRT

(Only supported for NVIDIA-GPUs, Tested on Linux Devices, Partial Dynamic Support)

You can convert ONNX weights to TensorRT by using the `convert.py` file. Simple run the following command: 

```
python convert.py --weights yolov5s.engine --img-size 720 1080
```

1. By default the onnx model is converted to TensorRT engine with FP16 precision. To convert to TensorRT engine with FP32 precision use ```--fp32``` when running the above command.

2. If using default weights, you do not need to download the ONNX model as the script will download it.

3. If you want to build the engine with custom image size, pass `--img-size custom_img_size` to `convert.py`

4. If you want to build the engine for your custom weights, simply do the following:

    - [Train Yolov5 on your custom dataset](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
    - [Export Weights PyTorch weights to ONNX](https://github.com/ultralytics/yolov5/blob/master/export.py)

    Make sure you use the `---dynamic` flag while exporting your custom weights.

    ```bash
    python export.py --weights $PATH_TO_PYTORCH_WEIGHTS --dynamic --include onnx
    ```

    Now simply use `python convert.py --weights path_to_custom_weights.onnx`, and you will have a converted TensorRT engine. Also add ```--nc``` (number of classes) if your custom model has different number of classes than COCO(i.e. 80 classes). 
    
