import cv2
import numpy as np
import time
import sys
import psutil
import matplotlib.pyplot as plt
sys.path.append("D:/Università/SORT/sort")
from sort import Sort

# Try to import GPUtil for GPU monitoring
try:
    import GPUtil
    gpu_available = True
except ImportError:
    print("GPUtil not installed: GPU metric will not be measured.")
    gpu_available = False

def main():
    cap = cv2.VideoCapture("highway_from_drone.mp4")
    if not cap.isOpened():
        print("Error opening video.")
        return

    # Parameters for saving the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    output_filename = "output_sort_tracker_1.mp4"
    out = cv2.VideoWriter(output_filename, fourcc, input_fps, (width, height))

    # Background Subtractor (MOG2)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    # Initialize the SORT tracker
    tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

    # Pixel -> meter conversion factor
    pixel_to_meter = 0.16

    # Speed limit in km/h
    speed_limit = 55.0

    # Dictionaries to save positions, speeds, and buffer for average calculation
    prev_positions = {}  # obj_id -> (centroid, timestamp)
    prev_speeds = {}     # obj_id -> smooth_speed
    speed_buffers = {}   # obj_id -> (sum_speed, count, last_update)

    # Lists to save metrics
    fps_list = []
    cpu_usage_list = []
    mem_usage_list = []
    gpu_usage_list = []

    # Initialize CPU counter
    psutil.cpu_percent(interval=None)

    start_total = time.time()  # total start time

    while True:
        start_frame = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # 1) Apply background subtraction
        fgmask = backSub.apply(frame)

        # 2) Clean the mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        # 3) Find contours (moving blobs)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # discard blobs that are too small
                x, y, w, h = cv2.boundingRect(cnt)
                dets.append([x, y, x + w, y + h, 1.0])
        dets = np.array(dets)

        # 4) Update the SORT tracker with the new detections
        trackers = tracker.update(dets)
        # trackers returns an array where each row is [x1, y1, x2, y2, id]

        # 5) Calculate the speed for each object and draw the overlay
        for d in trackers:
            x1, y1, x2, y2, obj_id = d
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            obj_id = int(obj_id)

            # Calculate measured speed (without smoothing)
            if obj_id in prev_positions:
                prev_centroid, prev_time = prev_positions[obj_id]
                dt = current_time - prev_time
                if dt > 0:
                    displacement_px = np.linalg.norm(np.array([cx, cy]) - prev_centroid)
                    displacement_m = displacement_px * pixel_to_meter
                    measured_speed = (displacement_m / dt) * 3.6  # km/h
                else:
                    measured_speed = 0.0
            else:
                measured_speed = 0.0

            prev_positions[obj_id] = (np.array([cx, cy]), current_time)

            # Buffer for calculating the average every 0.5 seconds
            if obj_id not in speed_buffers:
                speed_buffers[obj_id] = (0.0, 0, current_time)
            sum_speed, count, last_update_time = speed_buffers[obj_id]
            sum_speed += measured_speed
            count += 1
            speed_buffers[obj_id] = (sum_speed, count, last_update_time)

            if current_time - last_update_time >= 0.5:
                avg_speed = sum_speed / count
                smooth_speed = avg_speed if obj_id not in prev_speeds else 0.5 * prev_speeds[obj_id] + 0.5 * avg_speed
                prev_speeds[obj_id] = smooth_speed
                speed_buffers[obj_id] = (0.0, 0, current_time)
            else:
                smooth_speed = prev_speeds.get(obj_id, measured_speed)

            # Determine the color based on the speed limit
            color = (0, 255, 0) if smooth_speed <= speed_limit else (0, 0, 255)
            text = f"{smooth_speed:.1f} km/h"
            cx_int, cy_int = int(cx), int(cy)

            # Draw the circle at the center of the blob
            cv2.circle(frame, (cx_int, cy_int), 4, (255, 0, 0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            box_x = max(cx_int - text_w // 2, 0)
            box_y = max(cy_int - 20 - text_h, 0)
            cv2.rectangle(frame, (box_x, box_y),
                          (box_x + text_w, box_y + text_h + baseline),
                          (0, 0, 0), -1)
            cv2.putText(frame, text, (box_x, box_y + text_h),
                        font, font_scale, color, thickness)

        cv2.imshow("SORT Tracking", frame)
        out.write(frame)

        if cv2.waitKey(42) & 0xFF == 27:
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

    # Plot with the metrics
    frames_range = range(len(fps_list))
    plt.figure(figsize=(12, 6))
    plt.plot(frames_range, fps_list, label="FPS", color="blue")
    plt.plot(frames_range, cpu_usage_list, label="CPU Usage (%)", color="red")
    plt.plot(frames_range, mem_usage_list, label="Memory Usage (%)", color="green")
    if gpu_available and gpu_usage_list:
        plt.plot(frames_range, gpu_usage_list, label="GPU Usage (%)", color="purple")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.title("Performance per Frame - Background Subtraction + SORT Tracker")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
