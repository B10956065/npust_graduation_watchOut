"""伊婷的™交通偵測回報系統　traffic detection and reporting system
NPUST MIS 資料探勘應用 期末專題　指導教授:蔡○○教授
B10956003　蕭○○
B10956026　謝○○
B10956029　陳○○
B10956065　陳○○"""
import numpy as np
import cv2
from ultralytics import YOLO
from shapely.geometry import Polygon


def draw_outline():
    model = YOLO("yolov8s-seg-all-twRoadV2i-e50b0.pt")
    video_path = "targetImage/targetVideo750.mp4"
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            masked_frame = frame.copy()
            mask_car = []
            results = model(frame, classes=[0])
            for result in results:
                mask_car = list(result.masks.xy)
            mask_car = [mxy for mxy in mask_car if len(mxy) > 3]

            zero_mask = np.zeros_like(frame)
            if len(mask_car) > 0:
                polygons = [Polygon(mxy) for mxy in mask_car]
                list_accident_area = []
                for i, polygon in enumerate(polygons):  # iterate
                    list_overlap = []
                    for j, other_polygon in enumerate(polygons[i + 1:], start=1):  # iterate
                        if polygon.intersects(other_polygon):  # if polygon and other polygon overlap
                            list_overlap.append(i)
                            list_overlap.append(i + j)
                            # red outline
                            cv2.polylines(zero_mask, [np.array(polygon.exterior.coords, np.int32)],
                                          isClosed=True, color=(0, 0, 255), thickness=3)
                            cv2.polylines(zero_mask, [np.array(other_polygon.exterior.coords, np.int32)],
                                          isClosed=True, color=(0, 0, 255), thickness=3)

                            # blue rectangle
                            min_x = min(polygon.bounds[0], other_polygon.bounds[0])
                            min_y = min(polygon.bounds[1], other_polygon.bounds[1])
                            max_x = max(polygon.bounds[2], other_polygon.bounds[2])
                            max_y = max(polygon.bounds[3], other_polygon.bounds[3])
                            cv2.rectangle(zero_mask, (int(min_x), int(min_y)), (int(max_x), int(max_y)),
                                          color=(255, 0, 0), thickness=4)

                            # save accident area: [[x1,y1],[x2,y2],n,id1,id2,frame]
                            list_accident_area.append([[min_x, min_y], [max_x, max_y], 0, i, i + j, None])
                    if i not in list_overlap:  # if there are no any overlapping -> green outline
                        # FIXME: there are still repeated drawing, but i give up
                        cv2.polylines(zero_mask, [np.array(polygon.exterior.coords, np.int32)],
                                      isClosed=True, color=(0, 255, 0), thickness=1)
                masked_frame = cv2.addWeighted(frame, 1, zero_mask, 1, 0)
            else:
                print("Pass: no any object detected this frame.")

            # Output final result on screen
            cv2.imshow("Outline", masked_frame)

            # Shutdown program if user press "q" key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Program manual shutdown")
                break
        else:
            print("Program automatic shutdown")
            break


if __name__ == "__main__":
    draw_outline()
