import torch
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit, QScrollArea, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import sys
import os
# 将包含 lib 的目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib')))
# 导入模型
from models.pose_higher_hrnet import PoseHigherResolutionNet
from pose_estimation import PoseEstimation


class FitnessEvaluationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("健身动作评估系统")
        self.resize(1000, 600)

        # 加载人体姿态估计模型
        self.pose_estimator = PoseEstimation(model_path='models/MGS_HigherHRnet.pth.tar')

        # 视频显示区域
        self.standard_label = QLabel("标准动作视频")
        self.standard_label.setFixedSize(520, 360)
        self.standard_label.setStyleSheet("border: 1px solid black;")
        
        self.evaluation_label = QLabel("待评估动作视频")
        self.evaluation_label.setFixedSize(520, 360)
        self.evaluation_label.setStyleSheet("border: 1px solid black;")

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.standard_label)
        video_layout.addWidget(self.evaluation_label)

        # 评分区域
        self.score_area = QTextEdit()
        self.score_area.setReadOnly(True)
        self.score_area.setStyleSheet("border: 1px solid black;")
        self.score_area.setFixedHeight(200)

        # 按钮区域
        self.load_standard_btn = QPushButton("导入标准视频")
        self.load_standard_btn.setFixedSize(140, 40)

        self.load_evaluation_btn = QPushButton("导入待评估视频")
        self.load_evaluation_btn.setFixedSize(140, 40)

        self.start_evaluation_btn = QPushButton("开始评估")
        self.start_evaluation_btn.setFixedSize(140, 40)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.load_standard_btn)
        button_layout.addWidget(self.load_evaluation_btn)
        button_layout.addWidget(self.start_evaluation_btn)
        button_layout.addStretch()

        # 评分+按钮整体布局
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.score_area)
        bottom_layout.addLayout(button_layout)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

        # 信号绑定
        self.load_standard_btn.clicked.connect(self.load_standard_video)
        self.load_evaluation_btn.clicked.connect(self.load_evaluation_video)
        self.start_evaluation_btn.clicked.connect(self.start_evaluation)

    def load_standard_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择标准动作视频", "", "视频文件 (*.mp4 *.avi)")
        if file_name:
            self.standard_video_path = file_name
            self.display_video_frame(self.standard_video_path, self.standard_label)

    def load_evaluation_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择待评估动作视频", "", "视频文件 (*.mp4 *.avi)")
        if file_name:
            self.evaluation_video_path = file_name
            self.display_video_frame(self.evaluation_video_path, self.evaluation_label)

    def display_video_frame(self, video_path, label):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # 关键点预测
                keypoints = self.pose_estimator.predict_keypoints(frame)
                
                # 绘制骨架
                frame_with_skeleton = self.pose_estimator.draw_skeleton(frame, keypoints)
                
                # 转换为QImage并显示
                frame_with_skeleton = cv2.cvtColor(frame_with_skeleton, cv2.COLOR_BGR2RGB)
                height, width, channel = frame_with_skeleton.shape
                bytes_per_line = channel * width
                q_image = QImage(frame_with_skeleton.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
                label.setPixmap(pixmap)
        cap.release()

    def start_evaluation(self):
        if hasattr(self, 'standard_video_path') and hasattr(self, 'evaluation_video_path'):
            self.score_area.append("评估开始...")
            
            # 提取标准视频和待评估视频的关键点
            standard_keypoints = self.pose_estimator.extract_keypoints_from_video(self.standard_video_path)
            evaluation_keypoints = self.pose_estimator.extract_keypoints_from_video(self.evaluation_video_path)
            
            # 比较并评分
            score = self.pose_estimator.compare_keypoints(standard_keypoints, evaluation_keypoints)
            
            self.score_area.append(f"评估分数: {score}")
            self.score_area.append("评估完成！")
        else:
            self.score_area.append("请先导入视频！")

if __name__ == "__main__":
    app = QApplication([])
    window = FitnessEvaluationApp()
    window.show()
    app.exec_()
