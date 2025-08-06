import os
import cv2
import numpy as np
from PIL import Image
import supervision as sv
from ultralytics import YOLO

class Annotator:
    def __init__(self):
        self.model = None
        self.tracker = sv.ByteTrack()
        self.trace_annotator = sv.TraceAnnotator()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.label_annotator_center = sv.LabelAnnotator(text_position=sv.Position.CENTER_OF_MASS)
        self.mask_annotator = sv.MaskAnnotator()
        self.edge_annotator = sv.EdgeAnnotator()
        self.vertex_annotator = sv.VertexAnnotator()
        self.image_tasks = ['image_det', 'image_seg']
        self.video_tasks = ['video_id_only', 'video_id_track', 'video_keypoint']
        self.task2callback = {
            'video_id_only': self.callback_video_id_only,
            'video_id_track': self.callback_video_id_track,
            'video_keypoint': self.callback_video_keypoint
        }

    def __call__(self, video_or_image_path, task, model, target_path=None):
        # 检查输入是否为numpy数组（来自server的情况）
        if isinstance(video_or_image_path, np.ndarray):
            # 直接使用numpy数组，不需要设置target_path
            self.model = model
            if task in self.image_tasks:
                annotated_image = self.image_annotate(video_or_image_path, task)
                self.model = None
                return annotated_image
            else:
                raise ValueError(f"任务 '{task}' 不支持numpy数组输入")
        
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

        # Image processing
        if task in self.image_tasks:
            image = cv2.imread(video_or_image_path)
            annotated_image = self.image_annotate(image, task)
            annotated_image = Image.fromarray(annotated_image)
            annotated_image.save(target_path)
        elif task in self.video_tasks:
            self.video_annotate(video_or_image_path, target_path, self.task2callback[task])
        self.model = None
        return target_path

    def image_annotate(self, image, mode):
        results = self.model(image)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        if mode == 'image_det':
            annotated_image = self.box_annotator.annotate(scene=image, detections=detections)
            annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections)
        elif mode == 'image_seg':
            annotated_image = self.mask_annotator.annotate(scene=image, detections=detections)
            annotated_image = self.label_annotator_center.annotate(scene=annotated_image, detections=detections)
        
        return annotated_image    

    def callback_video_id_only(self, frame: np.ndarray, _: int) -> np.ndarray:
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {class_name}"
            for class_name, tracker_id
            in zip(detections.data["class_name"], detections.tracker_id)
        ]

        annotated_frame = self.box_annotator.annotate(
            frame.copy(), detections=detections)
        return self.label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)

    def callback_video_id_track(self, frame: np.ndarray, _: int) -> np.ndarray:
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {class_name}"
            for class_name, tracker_id
            in zip(detections.data["class_name"], detections.tracker_id)
        ]

        annotated_frame = self.box_annotator.annotate(
            frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)
        return self.trace_annotator.annotate(
            annotated_frame, detections=detections)

    def callback_video_keypoint(self, frame: np.ndarray, _: int) -> np.ndarray:
        results = self.model(frame)[0]
        key_points = sv.KeyPoints.from_ultralytics(results)
        detections = key_points.as_detections()
        detections = self.tracker.update_with_detections(detections)

        annotated_frame = self.edge_annotator.annotate(
            frame.copy(), key_points=key_points)
        annotated_frame = self.vertex_annotator.annotate(
            annotated_frame, key_points=key_points)
        annotated_frame = self.box_annotator.annotate(
            annotated_frame, detections=detections)
        return self.trace_annotator.annotate(
            annotated_frame, detections=detections)
    
    def video_annotate(self, video_path, target_path, callback):
        sv.process_video(
            source_path=video_path,
            target_path=target_path,
            callback=callback
        )

if __name__ == '__main__':
    annotator = Annotator()

    # image tasks
    img_path = './street.jpg'
    model = YOLO("yolov8n.pt")
    annotator(img_path, 'image_det', model, target_path='output/det_res.jpg')
    model = YOLO("yolov8n-seg.pt")
    annotator(img_path, 'image_seg', model, target_path='output/seg_res.jpg')

    # video tasks
    video_path = 'people-walking.mp4'
    model = YOLO("yolov8n.pt")
    annotator(video_path, 'video_id_only', model, target_path='output/track_id_res.mp4')
    annotator(video_path, 'video_id_track', model, target_path='output/track_id_track_res.mp4')
    model = YOLO("yolov8m-pose.pt")
    annotator(video_path, 'video_keypoint', model, target_path='output/track_keypoint.mp4')