import random
import math
import cv2
import numpy as np

def is_inside_bounds(points, W, H):
    """检查所有点是否都在 0~W, 0~H 范围内"""
    return all(0 <= x <= W and 0 <= y <= H for x, y in points)

def generate_rotated_rectangles(B=10, L=50, output_file="rectangles.txt", visualize=True):
    W, H = 896, 896  # 全局范围
    min_len, max_len = 180, 268  # 长边范围
    min_wid, max_wid = 90, 135   # 短边范围

    img = np.ones((H, W, 3), dtype=np.uint8) * 255  # 白底图

    with open(output_file, "w") as f:
        rect_id = 1
        while rect_id <= B:  # 用 while 确保有效矩形数量
            # 随机长宽
            length = random.randint(min_len, max_len)
            width = random.randint(min_wid, max_wid)

            # 随机中心点
            cx = random.uniform(length/2, W - length/2)
            cy = random.uniform(width/2,  H - width/2)

            # 随机旋转角度（弧度）
            angle = random.uniform(0, 2*math.pi)

            # 矩形4个顶点(以中心为原点)
            corners = [
                (-length/2, -width/2),
                ( length/2, -width/2),
                ( length/2,  width/2),
                (-length/2,  width/2)
            ]

            # 旋转矩阵
            rotated_corners = []
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for x, y in corners:
                rx = x * cos_a - y * sin_a + cx
                ry = x * sin_a + y * cos_a + cy
                rotated_corners.append((rx, ry))

            # 打乱顶点顺序
            shuffled_points = rotated_corners.copy()
            random.shuffle(shuffled_points)

            # 随机选择一个点替换
            replace_idx = random.randint(0, 3)
            corner_pts = []
            corner_pts.append(rotated_corners[(replace_idx+1)%4])  # 保留下一个点
            corner_pts.append(rotated_corners[(replace_idx+2)%4])  # 保留下下个点   

            # 大概率与其他点共线
            if random.random() < 0.99:
                orig_index = rotated_corners.index(shuffled_points[replace_idx])
                p1 = rotated_corners[orig_index]
                p2 = rotated_corners[(orig_index + 1) % 4]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                length_edge = math.hypot(dx, dy)
                t = 1 + (random.uniform(-L, L) / length_edge)
                offset = random.uniform(8, 10) if random.random() < 0.5 else -random.uniform(8, 10)
                new_x = p1[0] + dx * t + offset
                new_y = p1[1] + dy * t + offset
            else:
                rand_angle = random.uniform(0, 2*math.pi)
                rand_radius = random.uniform(0, L)
                new_x = cx + rand_radius * math.cos(rand_angle)
                new_y = cy + rand_radius * math.sin(rand_angle)

            shuffled_points[replace_idx] = (new_x, new_y)

            # 检查原矩形顶点 & 替换后的顶点是否在范围内
            if not (is_inside_bounds(rotated_corners, W, H) and 
                    is_inside_bounds(shuffled_points, W, H)):
                continue  # 直接丢弃，重新生成

            # 生成 mask
            masks = [0, 0, 0, 0]
            masks[replace_idx] = 1

            # 写入文件格式: id cx cy corn1 corny1 cornx2 corny2 x1 y1 x2 y2 x3 y3 m1 m2 m3 m4
            cx_rand = cx + random.uniform(-20, 20)
            cy_rand = cy + random.uniform(-20, 20)
            # 写入文件
            line = [str(rect_id)]
            line.append(f"{cx_rand:.2f}")
            line.append(f"{cy_rand:.2f}")
            line.append(f"{corner_pts[0][0]:.2f}")
            line.append(f"{corner_pts[0][1]:.2f}")
            line.append(f"{corner_pts[1][0]:.2f}")
            line.append(f"{corner_pts[1][1]:.2f}")
            for x, y in shuffled_points:
                line.append(f"{x:.2f}")
                line.append(f"{y:.2f}")
            line.extend(str(m) for m in masks)
            f.write(" ".join(line) + "\n")

            if visualize:
                # 画原矩形（红色）
                orig_poly = np.array(rotated_corners, dtype=np.int32)
                cv2.polylines(img, [orig_poly], True, (0, 0, 255), 2)

                # 画打乱后的点（绿色）
                for px, py in shuffled_points:
                    cv2.circle(img, (int(px), int(py)), 3, (0, 255, 0), -1)
                
                cv2.circle(img, (int(cx_rand), int(cy_rand)), 3, (255, 255, 0), -1)

                # 被替换的点（蓝色）
                px, py = shuffled_points[replace_idx]
                cv2.circle(img, (int(px), int(py)), 5, (255, 0, 0), -1)
                cv2.line(img, (int(px), int(py)), (int(cx_rand), int(cy_rand)), (255, 0, 0), 1)

            rect_id += 1

    if visualize:
        cv2.imshow("Rotated Rectangles with Shuffled Points", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"生成完成: {output_file}")


# 示例
for idx in range(1):
    # output_file = f"rectangles_{idx+1}_tmp.txt"
    # generate_rotated_rectangles(B=10, L=60, output_file=output_file, visualize=True)
    
    output_file = f"rectangles_{idx+1}.txt"
    generate_rotated_rectangles(B=1000, L=60, output_file=output_file, visualize=False)
