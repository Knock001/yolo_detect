import os
import sys
import argparse
import time
import cv2
import numpy as np
import ncnn
import threading
import queue

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_param', help='Path to ncnn param file (example: "yolo11n.opt.param")', required=True)
parser.add_argument('--model_bin', help='Path to ncnn bin file (example: "yolo11n.opt.bin")', required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")', default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), otherwise default to 640x480', default="640x480")
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.', action='store_true')

args = parser.parse_args()

# Parse user inputs
model_param_path = args.model_param
model_bin_path = args.model_bin
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Check if model files exist
if not os.path.exists(model_param_path) or not os.path.exists(model_bin_path):
    print('ERROR: Model files are invalid or not found. Make sure the param and bin filenames were entered correctly.')
    sys.exit(0)

# Load ncnn model
net = ncnn.Net()
net.load_param(model_param_path)
net.load_model(model_bin_path)

# Parse input to determine source type
img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = True
resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video', 'usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 10
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video':
        cap_arg = img_source
    elif source_type == 'usb':
        cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
    cap.set(cv2.CAP_PROP_FPS, 10)
    print(f"Camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors (Tableau 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Multithreading for inference
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue()

def inference_thread():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        # Resize frame for YOLOv11 inference
        in_frame = cv2.resize(frame, (320, 320))
        # Convert frame to ncnn Mat
        in_mat = ncnn.Mat.from_pixels(in_frame, ncnn.Mat.PixelType.PIXEL_BGR, 320, 320)
        ex = net.create_extractor()
        ex.input("images", in_mat)
        out_mat = ncnn.Mat()
        ex.extract("output", out_mat)
        result_queue.put(out_mat)

# Start inference thread
threading.Thread(target=inference_thread, daemon=True).start()

# Begin inference loop
while True:
    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    elif source_type == 'usb':
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. Exiting program.')
            break
    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if frame is None:
            print('Unable to read frames from the Picamera. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Add frame to queue for inference
    if frame_queue.full():
        frame_queue.get()  # Remove oldest frame if queue is full
    frame_queue.put(frame)

    # Process results from inference thread
    if not result_queue.empty():
        out_mat = result_queue.get()
        # Process ncnn output (example, adjust based on your ncnn model output format)
        # Placeholder for processing ncnn output (bounding boxes, classes, confidences)
        # Here, we assume output format is [num_detections, (x, y, w, h, conf, class_id)]
        detections = np.array(out_mat)  # Placeholder, replace with actual processing
        object_count = 0

        for detection in detections:
            xmin, ymin, w, h, conf, class_id = detection[:6]
            if conf > min_thresh:
                xmax = xmin + w
                ymax = ymin + h
                xmin, ymin, xmax, ymax = map(int, [xmin * resW / 320, ymin * resH / 320, xmax * resW / 320, ymax * resH / 320])
                color = bbox_colors[int(class_id) % 10]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f'Class {int(class_id)}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                object_count += 1

                # Calculate distance (example, adjust based on your focal length and object size)
                focal_length = 500  # Placeholder, adjust based on camera calibration
                real_object_size = 1.7  # Example: height of a person in meters
                bbox_height = ymax - ymin
                distance = (focal_length * real_object_size) / bbox_height
                print(f"Distance to object (class {int(class_id)}): {distance:.2f} meters")

    # Optional: Display frame (comment out for maximum FPS)
    # cv2.imshow('YOLO detection results', frame)
    if record:
        recorder.write(frame)

    # Wait for keypress
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('capture.png', frame)

    # Calculate FPS
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
frame_queue.put(None)  # Signal inference thread to stop
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()