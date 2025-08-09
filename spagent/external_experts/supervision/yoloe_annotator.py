import os
import cv2
import numpy as np
from PIL import Image
import supervision as sv
from ultralytics import YOLO, YOLOE

class Annotator:
    def __init__(self):
        self.model = None
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.color_annotator = sv.ColorAnnotator()

    def __call__(self, video_or_image_path, model, task='image', target_path=None):
        # 检查输入是否为numpy数组（来自server的情况）
        if isinstance(video_or_image_path, np.ndarray):
            # 直接使用numpy数组，不需要设置target_path
            self.model = model
            annotated_image = self.image_annotate(video_or_image_path)
            self.model = None
            return annotated_image
        
        # 原有的文件路径处理逻辑
        if target_path == None:
            target_dir, target_name = os.path.split(video_or_image_path)
            target_path = os.path.join(target_dir, 'output', target_name)
            if not os.path.exists(os.path.join(target_dir, 'output')):
                os.makedirs(os.path.join(target_dir, 'output'))
        else:
            target_dir, target_name = os.path.split(target_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

        self.model = model

        # 处理逻辑
        if task == 'image':
            # Image processing
            image = cv2.imread(video_or_image_path)
            annotated_image = self.image_annotate(image)
            annotated_image = Image.fromarray(annotated_image)
            annotated_image.save(target_path)
        elif task == 'video':
            # Video processing - 参考yoloe_test.py的方式
            self.video_annotate(video_or_image_path, target_path)
        
        self.model = None
        return target_path

    def image_annotate(self, image):
        results = self.model.predict(image, conf=0.3, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        annotated_image = self.box_annotator.annotate(scene=image, detections=detections)
        annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections)
        
        return annotated_image

    def video_annotate(self, video_path, target_path):
        """基于yoloe_test.py的视频处理方式"""
        from tqdm import tqdm
        
        frame_generator = sv.get_video_frames_generator(video_path)
        video_info = sv.VideoInfo.from_video_path(video_path)

        with sv.VideoSink(target_path, video_info) as sink:
            for index, frame in enumerate(tqdm(frame_generator, desc="Processing video")):
                results = self.model.predict(frame, conf=0.1, verbose=False)
                detections = sv.Detections.from_ultralytics(results[0])

                annotated_frame = frame.copy()
                annotated_frame = self.color_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections)

                sink.write_frame(annotated_frame)

if __name__ == '__main__':
    annotator = Annotator()

    # image tasks
    img_path = 'assets/dog.jpeg'
    names = ["dog", "eye", "tongue", "nose", "ear"]
    model = YOLOE("yoloe-v8l-seg.pt")
    model.set_classes(names, model.get_text_pe(names))
    annotator(img_path, model, task='image', target_path='outputs/det_dog.jpg')

    # video tasks - 重新创建模型实例避免状态冲突
    video_path = '/home/ubuntu/projects/spagent/assets/suitcases-1280x720.mp4'
    v_names = ["suitcase"]
    video_model = YOLOE("yoloe-v8l-seg.pt")  # 创建新的模型实例
    video_model.set_classes(v_names, video_model.get_text_pe(v_names))
    annotator(video_path, video_model, task='video', target_path='outputs/track_suitcases.mp4')