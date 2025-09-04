

import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog
from collections import OrderedDict
from typing import Dict, Tuple, List

# ------------------------------
# Configuration
# ------------------------------
OUTPUT_PATH = r"D:\\Python files\\OpenCv\\tracked.mp4"   # where output will be saved
SHOW_WINDOW = True   # Set to False if you don’t want video window

# ------------------------------
# Centroid Tracker
# ------------------------------
class CentroidTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: int = 80):
        self.next_object_id = 1
        self.objects: Dict[int, Tuple[int, int]] = OrderedDict()
        self.disappeared: Dict[int, int] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Tuple[int, int]):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, input_centroids: List[Tuple[int, int]]):
        if len(input_centroids) == 0:
            to_remove = []
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    to_remove.append(object_id)
            for oid in to_remove:
                self.deregister(oid)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.linalg.norm(np.array(object_centroids)[:, None, :] - np.array(input_centroids)[None, :, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])) - used_rows
        unused_cols = set(range(0, len(input_centroids))) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects

# ------------------------------
# Utility functions
# ------------------------------
def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    _, bin_mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    opened = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    return closed

def find_vehicle_boxes(mask: np.ndarray, min_area: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
    return boxes

def draw_boxes_and_ids(frame, boxes, objects, line_y, count_state):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for object_id, centroid in objects.items():
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(frame, f"ID {object_id}", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if count_state is not None:
            last_y = count_state['last_y'].get(object_id, cy)
            if last_y < line_y <= cy:
                count_state['down'] += 1
            elif last_y > line_y >= cy:
                count_state['up'] += 1
            count_state['last_y'][object_id] = cy

    if line_y is not None:
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)
        if count_state is not None:
            cv2.putText(frame, f"UP: {count_state['up']}  DOWN: {count_state['down']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

# ------------------------------
# Main processing
# ------------------------------
def process(video_path: str, output_path: str = None, show: bool = True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        sys.exit(1)

    writer = None
    backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    tracker = CentroidTracker(max_disappeared=25, max_distance=60)

    line_y = None
    count_state = {'up': 0, 'down': 0, 'last_y': {}}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if line_y is None:
            h, w = frame.shape[:2]
            line_y = int(h * 0.55)
            min_area = int((w * h) * 0.0008)

        fg_mask = backsub.apply(frame)
        clean = preprocess_mask(fg_mask)
        boxes = find_vehicle_boxes(clean, min_area=min_area)
        centroids = [(x + w // 2, y + h // 2) for (x, y, w, h) in boxes]
        objects = tracker.update(centroids)

        draw_boxes_and_ids(frame, boxes, objects, line_y, count_state)

        if output_path and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

        if writer is not None:
            writer.write(frame)

        if show:
            cv2.imshow("Vehicle Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

# ------------------------------
# Run program with file upload
# ------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide tkinter window
    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    if video_path:
        process(video_path, OUTPUT_PATH, show=SHOW_WINDOW)
    else:
        print("No video selected!")


