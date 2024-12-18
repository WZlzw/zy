import torch
import cv2
import numpy as np

import sys
import os

# 将包含 lib 的目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib')))
from models.pose_higher_hrnet import PoseHigherResolutionNet


class PoseEstimation:
    def __init__(self, model_path):
        # 加载模型
        self.model = PoseHigherResolutionNet()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()  # 设置模型为评估模式

    def predict_keypoints(self, frame):
        # 将图像转换为模型输入格式（假设模型需要调整大小，归一化等）
        input_image = self.prepare_image(frame)
        
        # 进行关键点预测
        with torch.no_grad():
            keypoints = self.model(input_image)  # 假设模型输出是关键点
        return keypoints.squeeze(0).numpy()  # 返回关键点的数组

    def prepare_image(self, frame):
        # 这里对图像进行预处理，调整尺寸和归一化
        frame_resized = cv2.resize(frame, (224, 224))
        frame_normalized = frame_resized / 255.0
        frame_tensor = torch.tensor(frame_normalized).permute(2, 0, 1).unsqueeze(0).float()
        return frame_tensor

    def draw_skeleton(self, frame, keypoints):
        # 假设keypoints是一个形如[(x1, y1), (x2, y2), ...]的列表
        for i in range(len(keypoints)):
            x, y = keypoints[i]
            if x > 0 and y > 0:  # 只绘制有效的关键点
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        # 连接骨架的部分（具体根据关键点的连接顺序进行设置）
        skeleton_connections = [(0, 1), (1, 2), (2, 3)]  # 示例连接，你需要根据模型输出的关键点定义连接
        for start, end in skeleton_connections:
            if keypoints[start][0] > 0 and keypoints[start][1] > 0 and keypoints[end][0] > 0 and keypoints[end][1] > 0:
                cv2.line(frame, (int(keypoints[start][0]), int(keypoints[start][1])),
                         (int(keypoints[end][0]), int(keypoints[end][1])), (0, 255, 0), 2)
        return frame

    def extract_keypoints_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        keypoints_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            keypoints = self.predict_keypoints(frame)
            keypoints_list.append(keypoints)
        cap.release()
        return keypoints_list

    def compare_keypoints(self, standard_keypoints, evaluation_keypoints):
        # 计算关键点之间的差异并返回评分
        score = 0
        for std_kps, eval_kps in zip(standard_keypoints, evaluation_keypoints):
            # 计算每一帧的差异，可以基于欧氏距离、角度等
            score += np.linalg.norm(np.array(std_kps) - np.array(eval_kps))  # 示例: 欧氏距离
        return max(0, 100 - score)  # 简单的评分逻辑，分数越高越好
