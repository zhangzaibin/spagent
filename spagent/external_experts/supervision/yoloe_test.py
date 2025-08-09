import supervision as sv
from ultralytics import YOLOE
from PIL import Image

IMAGE_PATH = "/home/ubuntu/projects/spagent/dataset/BLINK/ebb9c1c41b0fe3ff0d65cfc4ef3e2d26e4aefba3be654213a2aeab56d6546443.jpg"
NAMES = ["cat", "car"]

model = YOLOE("yoloe-v8l-seg.pt").cuda()
model.set_classes(NAMES, model.get_text_pe(NAMES))

image = Image.open(IMAGE_PATH)
results = model.predict(image, conf=0.3, verbose=False)

detections = sv.Detections.from_ultralytics(results[0])

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(scene=annotated_image, detections=detections)
annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)
print(1)

# import supervision as sv
# from ultralytics import YOLOE
# from PIL import Image
# from tqdm import tqdm

# SOURCE_VIDEO_PATH = "./assets/suitcases-1280x720.mp4"
# TARGET_VIDEO_PATH = "suitcases-1280x720-result.mp4"
# NAMES = ["suitcase"]

# model = YOLOE("yoloe-v8l-seg.pt").cuda()
# model.set_classes(NAMES, model.get_text_pe(NAMES))

# frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# # visualize video frames sample in notebook
# frames = []
# frame_interval = 10

# with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
#     for index, frame in enumerate(tqdm(frame_generator)):
#         results = model.predict(frame, conf=0.1, verbose=False)
#         detections = sv.Detections.from_ultralytics(results[0])

#         annotated_image = frame.copy()
#         annotated_image = sv.ColorAnnotator().annotate(scene=annotated_image, detections=detections)
#         annotated_image = sv.BoxAnnotator().annotate(scene=annotated_image, detections=detections)
#         annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)

#         sink.write_frame(annotated_image)

#         # visualize video frames sample in notebook
#         if index % frame_interval == 0:
#             frames.append(annotated_image)