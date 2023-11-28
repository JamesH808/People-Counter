import cv2
from ultralytics import YOLO
import supervision as sv
# import json

START = sv.Point(640,0)
END = sv.Point(640,720)

def main():
    model = YOLO("yolov8l.pt")
    line_zone = sv.LineZone(start = START, end = END)
    line_zone_annotator = sv.LineZoneAnnotator(thickness = 2, text_thickness = 1, text_scale =.5)

    box_annotator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness=1,
        text_scale=.5
    )

    for result in model.track(source=0, show=True, stream=True):
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.class_id == 0]
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        labels = [
            f"#{tracker_id}{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id
            in detections
        ]

     

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        line_zone.trigger(detections=detections)
        line_zone_annotator.annotate(frame=frame, line_counter = line_zone)

        cv2.imshow("yolov8", frame)

        if(cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()