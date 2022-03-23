import os
from imageai.Detection.Custom import (
    CustomObjectDetection,
    CustomVideoObjectDetection
)


def train_detection_model():
    from imageai.Detection.Custom import DetectionModelTrainer

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="fire-dataset")
    trainer.setTrainConfig(object_names_array=["fire"], batch_size=8, num_experiments=100,
                           train_from_pretrained_model="pretrained-yolov3.h5")
    # download 'pretrained-yolov3.h5' from the link below
    # https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5

    trainer.trainModel()


def detect_from_image(
    model_path='/models/detection_model-ex-33--loss-4.97.h5',
    config_path='./firenet/detection_config.json',
    input_img_path='data/1.jpg',
    output_img_path='data/1-detected.jpg'
):
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=model_path)
    detector.setJsonPath(configuration_json=config_path)
    detector.loadModel()

    detections = detector.detectObjectsFromImage(input_image=input_img_path,
                                                 output_image_path=output_img_path,
                                                 minimum_percentage_probability=40)

    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

def detect_from_video(
    model_path='./models/detection_model-ex-33--loss-4.97.h5',
    config_path='./firenet/detection_config.json',
    input_file_path='./data/video1.mp4',
    output_file_path='./data/video1-detected.mp4'

):
    detector = CustomVideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=model_path)
    detector.setJsonPath(configuration_json=config_path)
    detector.loadModel()

    detected_video_path = detector.detectObjectsFromVideo(
        input_file_path=input_file_path, 
        frames_per_second=30, 
        output_file_path=output_file_path, 
        minimum_percentage_probability=40, 
        log_progress=True
    )


if __name__ == '__main__':
    detect_from_image(
        model_path='./models/detection_model-ex-33--loss-4.97.h5',
        config_path='./firenet/detection_config.json',
        input_img_path='./data/1.jpg',
        output_img_path='./output/1-detected.jpg'
    )

    detect_from_video(
        model_path='./models/detection_model-ex-33--loss-4.97.h5',
        config_path='./firenet/detection_config.json',
        input_file_path='./data/video1.mp4',
        output_file_path='./output/video1-detected'
    )
