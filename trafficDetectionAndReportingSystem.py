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


def image_cutting(x1: int, y1: int, x2: int, y2: int, image):
    padding = 0
    height, width = image.shape[:2]
    x1 = min(0, x1 - padding)
    y1 = min(0, y1 - padding)
    x2 = max(height, x2 + padding)
    y2 = max(width, y2 + padding)
    result = image[y1:y2, x1:x2]
    return result, [[x1, y1], [x2, y2]]


def draw_outline():
    model = YOLO("yolov8s-seg-all-twRoadV2i-e50b0.pt")
    video_path = "targetImage/targetVideo750.mp4"
    list_accident_area = list()  # storge accident area
    list_waiting_delete = list()

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
                            min_x = int(min(polygon.bounds[0], other_polygon.bounds[0]))
                            min_y = int(min(polygon.bounds[1], other_polygon.bounds[1]))
                            max_x = int(max(polygon.bounds[2], other_polygon.bounds[2]))
                            max_y = int(max(polygon.bounds[3], other_polygon.bounds[3]))
                            cv2.rectangle(zero_mask, (min_x, min_y), (max_x, max_y),
                                          color=(255, 0, 0), thickness=4)

                            # save accident area: [x1,y1,x2,y2,n,id1,id2,frame]
                            accident_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            accident_frame, _ = image_cutting(min_x, min_y, max_x, max_y, accident_frame)
                            list_accident_area.append([min_x, min_y, max_x, max_y, 0, i, i + j, accident_frame])
                    if i not in list_overlap:  # if there are no any overlapping -> green outline
                        # FIXME: there are still repeated drawing
                        cv2.polylines(zero_mask, [np.array(polygon.exterior.coords, np.int32)],
                                      isClosed=True, color=(0, 255, 0), thickness=1)
                masked_frame = cv2.addWeighted(frame, 1, zero_mask, 1, 0)
            else:
                print("Pass: no any object detected this frame.")

            # TODO: monitor "accident_area"
            # list_accident_area = [x1, y1, x2, y2, n, id1, id2, frame]
            for i, acc in enumerate(list_accident_area):
                # get new frame
                old_frame = acc[-1].copy()
                new_frame, _ = image_cutting(*acc[:4], cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                list_accident_area[i][-1] = new_frame

                # FIXME: Optical flow is too slow, need another way to check if moving
                # optical flow
                flow = cv2.calcOpticalFlowFarneback(old_frame, new_frame, None, 0.8, 3, 5, 2, 5, 1.2, 0)
                flow = abs(flow.mean())

                # check if move
                if flow > 0.5:  # TODO: Adjustment the threshold of moving
                    # TODO: if moved => confirmed accident occurred
                    pass
                else:
                    # if NOT move => n+=1 and wait next round
                    list_accident_area[i][4] += 1

                # check life and add too old to list_waiting_delete
                if acc[4] > 30:  # TODO: Adjustment the threshold of kill acc area
                    list_waiting_delete.append(i)

            # pop item according to list_waiting_pop
            if len(list_waiting_delete):
                list_waiting_delete.sort(reverse=True)
                for i in list_waiting_delete:
                    list_accident_area.pop(i)
            list_waiting_delete.clear()

            # Output final result on screen
            cv2.imshow("Outline", masked_frame)

            # Shutdown program if user press "q" key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Program manual shutdown by \"q\" key pressed")
                break
        else:
            print("Program automatic shutdown")
            break


if __name__ == "__main__":
    draw_outline()
