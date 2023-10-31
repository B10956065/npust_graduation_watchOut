from ultralytics import YOLO
import cv2
import numpy as np
import math
from AdaBins_main.infer import InferenceHelper
from PIL import Image


def initialize(yolo_model: str = "yolov8s-seg-all-twRoadV2i-e50b0.pt",
               depth_model: str = "kitti"):
    print("Hello World! EVERYTHING ON the ROAD IN the VIDEO")
    model = YOLO(yolo_model)
    infer_helper = InferenceHelper(dataset=depth_model)
    return model, infer_helper


def main(video_frame, yolo_model, depth_model, carame_setting: int = 1):
    # {0: 'Car', 1: 'Crosswalk', 2: 'Foot walk', 3: 'Human', 4: 'Motorcycle',
    #  5: 'Road', 6: 'Road_line', 7: 'Truck', 8: 'traffic_light'}
    frame = video_frame
    results = yolo_model(source=frame, stream=True, device=0)
    for result in results:
        # extract mask of car and human
        mask_car = []
        mask_human = []
        className = result.names
        classId = result.boxes.cls.tolist()
        list_xy = result.masks.xy
        for iid in range(len(classId)):
            iid = int(iid)
            if classId[iid] in [0, 4]:
                if len(list_xy[int(iid)]) >= 3:
                    mask_car.append(list_xy[int(iid)])
            elif classId[iid] in [3]:  # FIXME: Adjustment back to correct class id after test completed
                if len(list_xy[int(iid)]) >= 3:
                    mask_human.append(list_xy[int(iid)])

        # visualization mask of car and human
        zeroMask = np.zeros_like(frame)
        flag_mask = 0
        for mxy in mask_car:
            mxy = np.array(mxy)
            cv2.fillPoly(zeroMask, np.int32([mxy]), color=(255, 0, 0))
            flag_mask += 1
        for mxy in mask_human:
            mxy = np.array(mxy)
            cv2.fillPoly(zeroMask, np.int32([mxy]), color=(0, 255, 0))
        maskedFrame = cv2.addWeighted(frame, 0.7, zeroMask, 0.3, 0)

        # Estimation depth
        batch = cv2.cvtColor(cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        _, depth_map = depth_model.predict_pil(Image.fromarray(batch))
        depth_map = cv2.resize(depth_map[0, 0], (1280, 720), interpolation=cv2.INTER_NEAREST)
        depth_map = np.expand_dims(depth_map, axis=0)

        # visualization depth of car and human
        depth_car = []
        depth_human = []
        depthMask = np.zeros_like(frame)
        for mask in mask_car:
            # mxy = np.array(mask)
            # cv2.fillPoly(depthMask, np.int32([mxy]), color=(255, 0, 0))

            # 將座標轉換為整數並選出對應的深度值
            mask_int = mask.astype(int)
            depths = depth_map[0][mask_int[:, 1], mask_int[:, 0]]

            # 計算平均深度並將結果加入列表
            if len(depths) > 0:
                average_depth = np.mean(depths)
                depth_car.append(average_depth.item())
            else:
                depth_car.append(0.0)  # 如果沒有對應的深度值，將平均深度設置為0.0
                continue

            # cv2.putText(maskedFrame, f"{ound(depth_car[-1], 2)}", (mask_int[0][0], mask_int[0][1]),
            #             cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(150, 150, 255))

        for mask in mask_human:
            mxy = np.array(mask)
            # cv2.fillPoly(depthMask, np.int32([mxy]), color=(0, 255, 0))

            # 將座標轉換為整數並選出對應的深度值
            mask_int = mask.astype(int)
            depths = depth_map[0][mask_int[:, 1], mask_int[:, 0]]

            # 計算平均深度並將結果加入列表
            if len(depths) > 0:
                average_depth = np.mean(depths)
                depth_human.append(average_depth.item())
            else:
                depth_human.append(0.0)  # 如果沒有對應的深度值，將平均深度設置為0.0

            # cv2.putText(maskedFrame, f"{round(depth_human[-1], 2)}", (mask_int[0][0], mask_int[0][1]),
            #             cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(255, 255, 150))
            flag_mask += 1

        def polygon_centroid(vertices):  # 多邊形重心
            num_vertices = len(vertices)

            if num_vertices < 3:
                raise ValueError("A polygon must have at least 3 vertices.")

            centroid = np.array([0.0, 0.0])
            signed_area = 0.0

            for i in range(num_vertices):
                x0, y0 = vertices[i]
                x1, y1 = vertices[(i + 1) % num_vertices]

                cross_product = (x0 * y1 - x1 * y0)
                signed_area += cross_product

                centroid[0] += (x0 + x1) * cross_product
                centroid[1] += (y0 + y1) * cross_product

            signed_area *= 0.5
            centroid /= (6 * signed_area)
            return centroid

        # 相機校準參數 (焦距、影像中心等)
        dict_sig = {1: (1191, 1225, 6), 2: (1191, 1225, 3), 3: (380, 680, 15), 4: (380, 680, 15), 5: (380, 680, 12)}
        # TODO: 上面的參數，括號內第三個是過近的距離
        fx, fy, fd = dict_sig[carame_setting]
        cx = 720.0  # 影像中心 x 座標
        cy = 360.0  # 影像中心 y 座標

        # Centroid
        centroid_car = []
        centroid_human = []
        flag_centroid = 0
        for llist in [mask_car, mask_human]:
            for mxy in llist:
                centroid = polygon_centroid(mxy)
                if flag_centroid == 0:
                    centroid_car.append(centroid)
                elif flag_centroid == 1:
                    centroid_human.append(centroid)
            flag_centroid += 1

        # 計算車和人之間的夾角，並推算距離
        list_re = list()
        flag_cc = 0  # car
        for cc in centroid_car:
            flag_ch = 0  # human
            for ch in centroid_human:
                # 計算夾角
                pixel_a = cc  # [x, y] 座標
                pixel_b = ch

                # 將像素座標轉換為攝影機座標（根據相機校準參數）
                camera_point_a = np.array([(pixel_a[0] - cx) / fx, (pixel_a[1] - cy) / fy, 1.0])
                camera_point_b = np.array([(pixel_b[0] - cx) / fx, (pixel_b[1] - cy) / fy, 1.0])

                # 計算兩個向量之間的夾角（使用點積）
                dot_product = np.dot(camera_point_a, camera_point_b)
                norm_a = np.linalg.norm(camera_point_a)
                norm_b = np.linalg.norm(camera_point_b)
                angle_radians = np.arccos(dot_product / (norm_a * norm_b))

                # 將弧度轉換為度數
                angle_degrees = np.degrees(angle_radians)

                # 計算距離
                hornBAC = angle_degrees
                lineAB = depth_car[flag_cc]
                lineAC = depth_human[flag_ch]

                # 角度轉弧度
                hornBAC_rad = math.radians(hornBAC)

                # 使用餘弦定理計算lineBC的長度
                BC_squared = lineAB ** 2 + lineAC ** 2 - 2 * lineAB * lineAC * math.cos(hornBAC_rad)
                lineBC = math.sqrt(BC_squared)

                if flag_ch:
                    tt = tuple(int(x) for x in cc)

                    # cv2.putText(maskedFrame, f"{lineBC:.2f}", tt,
                    #            cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(10, 10, 10))

                print(f"車{flag_cc}與人{flag_ch}的距離: {lineBC:.2f} 單位")

                if lineBC <= fd:  # FIXME: Adjustment distance
                    list_re.append(f"車{flag_cc}與人{flag_ch}的距離過近")
                    # Find the minimum and maximum x, y coordinates
                    min_x = np.min(mask_car[flag_cc][:, 0])
                    max_x = np.max(mask_car[flag_cc][:, 0])
                    min_y = np.min(mask_car[flag_cc][:, 1])
                    max_y = np.max(mask_car[flag_cc][:, 1])

                    # Draw the bounding box
                    bounding_box_color = (255, 255, 255)  # Red color
                    cv2.rectangle(maskedFrame, (int(min_x), int(min_y)), (int(max_x), int(max_y)),
                                  bounding_box_color, 3)
                else:
                    list_re.append(f"車{flag_cc}與人{flag_ch}的距離正常")
                    pass

                flag_ch += 1
            flag_cc += 1

        # show
        return maskedFrame, list_re
