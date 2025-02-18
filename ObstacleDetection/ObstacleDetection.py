import sys
import cv2
import os
import math
import numpy as np
import threading
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from filterpy.kalman import KalmanFilter
import time
import logging 
from datetime import datetime
import csv


class ObstacleLogger:
    def __init__(self, log_file="engel_tespiti_log.csv"):
        self.log_file = log_file
        self.last_obstacle_time = None
        self.initialize_log_file()

    def initialize_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'Tarih_Saat',
                    'Olay_Tipi',
                    'Tespit_Edilen_Anahtar_Nokta',
                    'Engel_Tespit_Zamani',
                    'Uyari_Zamani',
                    'Gecikme_Suresi(sn)',
                    'Aci_Degisimi(derece)',
                    'Aciklama'
                ])
            print(f"Log dosyası oluşturuldu: {self.log_file}")

    def format_time(self, timestamp):
        """Zamanı formatlı string olarak döndürür"""
        if timestamp is None:
            return "N/A"
        return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")[:-4]

    def log_event(self, event_type, keypoint_count, detection_time, warning_time, angle_change=None):
        current_time = time.time()

        # Gecikme süresini hesapla (saniye cinsinden)
        if warning_time and detection_time:
            delay = warning_time - detection_time
        else:
            delay = 0

        # Olay tipine göre açıklama oluştur
        if event_type == "ENGEL_TESPIT":
            description = f"Engel tespit edildi - Anahtar nokta sayısı: {keypoint_count}"
            self.last_obstacle_time = detection_time
        elif event_type == "ACI_UYARI":
            description = f"Açı değişimi tespit edildi: {angle_change:.2f}° - Eşik değeri aşıldı"
        else:
            description = "Bilinmeyen olay"

        with open(self.log_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                event_type,
                keypoint_count,
                self.format_time(detection_time),
                self.format_time(warning_time),
                f"{delay:.3f}",
                f"{angle_change:.2f}" if angle_change is not None else "N/A",
                description
            ])

        print(f"\nYeni Olay Kaydedildi:")
        print(f"Zaman: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Olay: {event_type}")
        print(f"Açıklama: {description}")
        if delay > 0:
            print(f"Gecikme Süresi: {delay:.3f} saniye")
        print("-" * 50)

class AdvancedAngleDetection:
    def __init__(self, num_regions=4):
        self.num_regions = num_regions
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([0., 0.])  # initial state (angle and angular velocity)
        self.kf.F = np.array([[1., 1.], [0., 1.]])  # state transition matrix
        self.kf.H = np.array([[1., 0.]])  # measurement function
        self.kf.P *= 1000.  # covariance matrix
        self.kf.R = 5  # measurement noise
        self.kf.Q = np.eye(2) * 0.1  # process noise

    def detect_angle_change(self, curr_frame, prev_frame):
        if prev_frame is None:
            return 0

        height, width = curr_frame.shape[:2]
        region_height = height // 2
        region_width = width // 2

        angles = []
        for i in range(self.num_regions):
            y = (i // 2) * region_height
            x = (i % 2) * region_width

            curr_roi = curr_frame[y:y + region_height, x:x + region_width]
            prev_roi = prev_frame[y:y + region_height, x:x + region_width]

            angle = self.calculate_affine_angle(prev_roi, curr_roi)
            if angle is not None:
                angles.append(angle)

        if angles:
            median_angle = np.median(angles)
            try:
                self.kf.predict()
                filtered_state = self.kf.update(np.array([median_angle]))
                if filtered_state is not None and filtered_state.size > 0:
                    filtered_angle = filtered_state[0, 0]
                    return filtered_angle
                else:
                    print("Kalman filter returned invalid state")
                    return median_angle
            except Exception as e:
                print(f"Error in Kalman filter: {e}")
                return median_angle
        else:
            print("No valid angles detected")
            return 0

    def calculate_affine_angle(self, img1, img2):
        try:
            # Convert images to grayscale if they're not already
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            if len(img2.shape) == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Create SIFT detector
            sift = cv2.SIFT_create()

            # Detect keypoints and compute descriptors
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                try:
                    # Create BFMatcher with default params
                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(des1, des2, k=2)

                    # Apply ratio test
                    good_matches = []
                    for match_group in matches:
                        if len(match_group) >= 2:  # Check if we have at least 2 matches
                            m, n = match_group
                            if m.distance < 0.75 * n.distance:
                                good_matches.append(m)

                    if len(good_matches) > 10:
                        # Extract location of good matches
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        # Find affine transformation
                        M, _ = cv2.estimateAffine2D(src_pts, dst_pts)

                        if M is not None:
                            # Calculate rotation angle
                            angle = -np.arctan2(M[0, 1], M[0, 0]) * 180 / np.pi
                            return angle

                except cv2.error as e:
                    print(f"OpenCV Error in calculate_affine_angle: {e}")
                    return None
                except Exception as e:
                    print(f"Error in calculate_affine_angle: {e}")
                    return None

            return None

        except Exception as e:
            print(f"General error in calculate_affine_angle: {e}")
            return None





class ObstacleDetection(QObject):
    UpdateTable = pyqtSignal()
    Finished = pyqtSignal()

    def __init__(self, detector_index=6, parent=None):
        super().__init__(parent)
        self.detectorIndex = detector_index
        self.maxKP = 0
        self.isFirstData = True
        self.isFinished = False

    def calculate_angle(self, p1, p2):
        delta_y = p2[1] - p1[1]
        delta_x = p2[0] - p1[0]
        angle = math.atan2(delta_y, delta_x) * (180 / math.pi)
        return angle

    def FeatureDetection(self, img):
        try:
            if self.detectorIndex == 0:
                featureDetector = cv2.SIFT_create()
            elif self.detectorIndex == 1:
                featureDetector = cv2.xfeatures2d.SURF_create()
            elif self.detectorIndex == 2:
                featureDetector = cv2.BRISK_create()
            elif self.detectorIndex == 3:
                featureDetector = cv2.ORB_create()
            elif self.detectorIndex == 4:
                featureDetector = cv2.KAZE_create()
            elif self.detectorIndex == 5:
                featureDetector = cv2.AKAZE_create()
            elif self.detectorIndex == 6:
                featureDetector = cv2.FastFeatureDetector_create()
                keypoints = featureDetector.detect(img, None)
                descriptors = None
                return keypoints, descriptors
            elif self.detectorIndex == 7:
                featureDetector = cv2.AgastFeatureDetector_create()
                keypoints = featureDetector.detect(img, None)
                descriptors = None
                return keypoints, descriptors
            elif self.detectorIndex == 8:
                featureDetector = cv2.MSER_create()
                keypoints = featureDetector.detect(img, None)
                descriptors = None
                return keypoints, descriptors
            elif self.detectorIndex == 9:
                featureDetector = cv2.ORB_create()
            else:
                featureDetector = cv2.SIFT_create()

            keypoints = featureDetector.detect(img, None)
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:200]

            keypoints = self.remove_duplicated_keypoints(keypoints)
            keypoints = self.filter_keypoints_by_border(keypoints, img.shape[:2], border=0)
            keypoints = self.filter_keypoints_by_size(keypoints, min_size=1.0, max_size=float('inf'))

            if hasattr(featureDetector, 'compute'):
                keypoints, descriptors = featureDetector.compute(img, keypoints)
            else:
                descriptors = None

            return keypoints, descriptors
        except Exception as e:
            print(f"Hata özellik tespitinde: {e}")
            return [], None

    def remove_duplicated_keypoints(self, keypoints, tolerance=1.0):
        unique_keypoints = []
        for kp in keypoints:
            if not any(self.keypoint_distance(kp, unique_kp) < tolerance for unique_kp in unique_keypoints):
                unique_keypoints.append(kp)
        return unique_keypoints



    def keypoint_distance(self, kp1, kp2):
        return math.hypot(kp1.pt[0] - kp2.pt[0], kp1.pt[1] - kp2.pt[1])

    def filter_keypoints_by_border(self, keypoints, image_shape, border=0):
        height, width = image_shape
        return [kp for kp in keypoints if border <= kp.pt[0] < width - border and border <= kp.pt[1] < height - border]

    def filter_keypoints_by_size(self, keypoints, min_size=1.0, max_size=float('inf')):
        return [kp for kp in keypoints if min_size <= kp.size <= max_size]

    def ProcessImage(self, img):
        edges = cv2.Canny(img, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)
                for i in range(len(contour) - 1):
                    angle = self.calculate_angle(contour[i][0], contour[i + 1][0])
                    print(f"Açı: {angle:.2f}°")

        keypoints, descriptors = self.FeatureDetection(img)

        self.maxKP = max(len(keypoints), self.maxKP)

        if self.isFirstData:
            self.isFirstData = False
        else:
            self.UpdateTable.emit()

        img = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

        return img, len(keypoints), keypoints, descriptors

class FastAlgorithmApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        rtsp_url = "rtsp://admin:Embedlab38.@30.10.23.17:554"
        self.capture = cv2.VideoCapture(rtsp_url)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        if not self.capture.isOpened():
            print("Kamera açılamadı")
            sys.exit()

        self.previous_frame = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.timer = QTimer()
        self.logger = ObstacleLogger()
        self.last_detection_time = None
        self.last_warning_time = None
        self.timer.timeout.connect(self.timerEvent)
        self.keypoint_threshold = 200
        self.rotation_angle = 0.0
        self.frame_count = 0
        # Uyarı mesajı için yeni değişkenler
        self.warning_start_time = None
        self.warning_duration = 5  # saniye cinsinden
        self.show_warning = False
        self.last_angle_change = 0.0
        self.obstacle_detector = ObstacleDetection(detector_index=6)
        self.obstacle_detector.UpdateTable.connect(self.update_table)
        self.obstacle_detector.Finished.connect(self.stop_detection)
        self.logger = ObstacleLogger()
        self.last_detection_time = None
        self.last_warning_time = None

        self.obstacle_detected = False

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.processing_thread = None
        self.processed_frame = None
        self.is_processing = False

        # ROI parametreleri
        self.roi_x = 50  # Sol kenardan başlangıç noktası
        self.roi_y = 50  # Üst kenardan başlangıç noktası
        self.roi_width = 540  # ROI genişliği
        self.roi_height = 380  # ROI yüksekliği
        self.angle_detector = AdvancedAngleDetection()

    def initUI(self):
        self.setWindowTitle('FAST Algoritması ile Engel Tespiti')
        self.layout = QVBoxLayout()

        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        self.detector_combo = QComboBox(self)
        self.detector_combo.addItems([
            "SIFT", "SURF", "BRISK", "ORB", "KAZE",
            "AKAZE", "FAST", "AGAST", "MSER", "ORB (TBMR)"
        ])
        self.detector_combo.currentIndexChanged.connect(self.change_detector)
        self.layout.addWidget(self.detector_combo)

        self.start_button = QPushButton('Başlat', self)
        self.start_button.clicked.connect(self.start_detection)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Durdur', self)
        self.stop_button.clicked.connect(self.stop_detection)
        self.layout.addWidget(self.stop_button)

        self.angle_label = QLabel("Rotation Angle: 0.0°", self)
        self.layout.addWidget(self.angle_label)

        self.setLayout(self.layout)

    def change_detector(self, index):
        self.obstacle_detector.detectorIndex = index
        print(f"Detektör değiştirildi: {self.detector_combo.currentText()} (Index: {index})")
        self.matcher = cv2.BFMatcher(cv2.NORM_L2 if index in [0, 1, 3, 4, 5, 7, 8, 9] else cv2.NORM_HAMMING, crossCheck=True)

    def start_detection(self):
        self.timer.start(10)
        print("Engel tespiti başlatıldı.")

    def stop_detection(self):
        self.timer.stop()
        if self.capture.isOpened():
            self.capture.release()
        cv2.destroyAllWindows()
        print("Engel tespiti durduruldu.")

    def timerEvent(self):
        ret, frame = self.capture.read()
        if not ret:
            print("Kamera görüntüsü alınamadı")
            return

        if not self.is_processing:
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self.process_frame, args=(frame,))
            self.processing_thread.start()

        if self.processed_frame is not None:
            self.display_frame(self.processed_frame)

    def process_frame(self, frame):
        frame = cv2.resize(frame, (640, 480))
        current_time = time.time()

        roi = frame[self.roi_y:self.roi_y + self.roi_height,
              self.roi_x:self.roi_x + self.roi_width]

        processed_roi, keypoint_count, keypoints, descriptors = self.obstacle_detector.ProcessImage(roi)

        frame[self.roi_y:self.roi_y + self.roi_height,
        self.roi_x:self.roi_x + self.roi_width] = processed_roi

        # Engel tespiti loglama
        if keypoint_count < self.keypoint_threshold:
            if not self.obstacle_detected:  # Sadece yeni engel tespitinde logla
                self.obstacle_detected = True
                self.last_detection_time = current_time

                # Engel tespiti olayını logla
                self.logger.log_event(
                    "ENGEL_TESPIT",
                    keypoint_count,
                    self.last_detection_time,
                    None
                )

            cv2.putText(frame, "Engel Tespit Edildi!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            self.obstacle_detected = False

        cv2.rectangle(frame, (self.roi_x, self.roi_y),
                      (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                      (0, 255, 0), 2)

        if self.previous_frame is not None:
            angle_change = self.angle_detector.detect_angle_change(frame, self.previous_frame)

            ANGLE_THRESHOLD = 5.0

            if abs(angle_change) > ANGLE_THRESHOLD:
                self.show_warning = True
                self.warning_start_time = current_time
                self.last_warning_time = current_time
                self.last_angle_change = angle_change

                # Açı değişimi uyarısını logla
                self.logger.log_event(
                    "ACI_UYARI",
                    keypoint_count,
                    self.last_detection_time if self.last_detection_time else current_time,
                    self.last_warning_time,
                    angle_change
                )

            if self.show_warning:
                if current_time - self.warning_start_time <= self.warning_duration:
                    warning_text = f"Aci Degisimi: "
                    cv2.putText(frame, warning_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    remaining_time = int(self.warning_duration -
                                         (current_time - self.warning_start_time))
                    time_text = f"({remaining_time}s)"
                    cv2.putText(frame, time_text, (300, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    self.show_warning = False

            self.rotation_angle += angle_change if angle_change else 0
            self.angle_label.setText(f"Rotation Angle: {self.rotation_angle:.2f}°")

        self.previous_frame = frame.copy()
        self.previous_keypoints = keypoints
        self.previous_descriptors = descriptors

        self.processed_frame = frame
        self.is_processing = False
        self.frame_count += 1



    def update_table(self):
        print("Tablo güncellendi.")

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        print("\nUygulama kapatılıyor...")
        print("Son loglar kaydediliyor...")
        self.timer.stop()
        if self.capture.isOpened():
            self.capture.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FastAlgorithmApp()
    ex.show()
    sys.exit(app.exec_())