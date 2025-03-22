import cv2
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
from ultralytics import YOLO
from norfair import Detection, Tracker

# Try to import GPUtil for GPU monitoring
try:
    import GPUtil
    gpu_available = True
except ImportError:
    print("GPUtil not installed: GPU metric will not be measured.")
    gpu_available = False

def detection_function(frame, model):
    # Perform detection with YOLOv8
    results = model(frame)
    detections = []
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box[:4]
            # Calculate the center as a 2D array (shape (1,2))
            centroid = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
            detections.append(Detection(points=centroid, scores=np.array([1.0])))
    return detections

def main():
    # Load the YOLOv8 model (nano version for speed)
    model = YOLO("yolov8n.pt")

    # Initialize the Norfair tracker with Euclidean distance
    tracker = Tracker(
        distance_function=lambda detection, tracked_obj: np.linalg.norm(
            detection.points - tracked_obj.last_detection.points
        ),
        distance_threshold=30
    )

    # Conversion factor: pixel -> meter
    pixel_to_meter = 0.16

    # Speed limit (km/h) for text color change
    speed_limit = 55.0

    # Dictionary to store the position and time of the last frame for each ID
    prev_positions = {}

    # Open the video file
    cap = cv2.VideoCapture("highway_from_drone.mp4")
    if not cap.isOpened():
        print("Error opening video.")
        return

    # Parameters for saving the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    output_filename = "output_yolo_norfair_tracker_1.mp4"
    out = cv2.VideoWriter(output_filename, fourcc, input_fps, (width, height))

    # Lists to save the metrics
    fps_list = []
    cpu_usage_list = []
    mem_usage_list = []
    gpu_usage_list = []

    # Initialize the CPU counter
    psutil.cpu_percent(interval=None)
    start_total = time.time()

    while True:
        start_frame = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # Get the detections from the YOLO model
        detections = detection_function(frame, model)

        # Update the tracker with the new detections
        tracked_objects = tracker.update(detections=detections)

        # For each tracked object, calculate the displacement and estimate speed
        for obj in tracked_objects:
            centroid = obj.last_detection.points[0]
            obj_id = obj.id

            if obj_id in prev_positions:
                prev_centroid, prev_time = prev_positions[obj_id]
                dt = current_time - prev_time
                if dt > 0:
                    displacement = np.linalg.norm(centroid - prev_centroid)
                    speed_m_s = (displacement * pixel_to_meter) / dt
                    speed_km_h = speed_m_s * 3.6
                else:
                    speed_km_h = 0.0
            else:
                speed_km_h = 0.0

            prev_positions[obj_id] = (centroid, current_time)

            # ID into a letter (A, B, C, ...), restarting from A if >26
            letter_label = chr(65 + (int(obj_id) % 26))
            text = f'{letter_label} {speed_km_h:.1f} km/h'
            text_color = (0, 255, 0) if speed_km_h <= speed_limit else (0, 0, 255)
            pos = (int(centroid[0]), int(centroid[1]))
            cv2.circle(frame, pos, 4, (0, 255, 0), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            rect_x = max(pos[0] - 20, 0)
            rect_y = max(pos[1] - 10 - text_h - baseline, 0)
            cv2.rectangle(frame, (rect_x, rect_y), (rect_x + text_w, rect_y + text_h + baseline), (0, 0, 0), -1)
            cv2.putText(frame, text, (rect_x, rect_y + text_h), font, font_scale, text_color, thickness)

        # Display the annotated frame and write it to the output video
        cv2.imshow("YOLOv8 + Norfair (Speed Estimation)", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Record the metrics for the frame
        elapsed = time.time() - start_frame
        fps_list.append(1.0 / elapsed if elapsed > 0 else 0)
        cpu_usage_list.append(psutil.cpu_percent(interval=None))
        mem_usage_list.append(psutil.virtual_memory().percent)
        if gpu_available:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_load = np.mean([gpu.load * 100 for gpu in gpus])
            else:
                gpu_load = 0
            gpu_usage_list.append(gpu_load)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_total
    print(f"Total execution time: {total_time:.2f} seconds")
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"Average FPS: {avg_fps:.2f}")

    # Create a plot with the metrics (FPS, CPU, Memory, GPU)
    frames_range = range(len(fps_list))
    plt.figure(figsize=(12, 6))
    plt.plot(frames_range, fps_list, label="FPS", color="blue")
    plt.plot(frames_range, cpu_usage_list, label="CPU Usage (%)", color="red")
    plt.plot(frames_range, mem_usage_list, label="Memory Usage (%)", color="green")
    if gpu_available and gpu_usage_list:
        plt.plot(frames_range, gpu_usage_list, label="GPU Usage (%)", color="purple")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.title("Performance per Frame - YOLOv8n + Norfair")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
