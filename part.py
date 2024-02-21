import cv2
import numpy as np
import time
import serial

ser = serial.Serial('/dev/ttyUSB0', 9600)

def detect_objects():
    net = cv2.dnn.readNet("yolov4-tiny-custom_last.weights", "yolov4-tiny-custom.cfg")
    output_layers = net.getUnconnectedOutLayersNames()
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    cap = cv2.VideoCapture(2)

    start_time = time.time()
    frame_count = 0

    part_depan_value = 10
    part_belakang_value = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        confidences = []
        boxes = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    # Mengirim nilai part_depan_value dan part_belakang_value ke Arduino
                    if classes[class_id] == "part depan":
                        if center_x > width // 2:
                            if center_x > width - 50:
                                part_depan_value = 10
                            else:
                                part_depan_value = 11
                        else:
                            part_depan_value = 10
                        ser.write(str(part_depan_value).encode())  # Mengirim nilai part_depan_value ke Arduino
                    elif classes[class_id] == "part belakang":
                        if center_x > width // 2:
                            if center_x > width - 50:
                                part_belakang_value = 20
                            else:
                                part_belakang_value = 21
                        else:
                            part_belakang_value = 20
                        ser.write(str(part_belakang_value).encode())  # Mengirim nilai part_belakang_value ke Arduino

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                label = f"{classes[class_id]}: {confidence:.2f}"
                
                x, y, w, h = box
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Calculate center coordinates of the bounding box
                center_x_box = x + w // 2
                center_y_box = y + h // 2

                # Draw center point
                cv2.circle(frame, (center_x_box, center_y_box), 5, (255, 0, 0), -1)

        # Draw vertical line in the middle of the frame
        cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 255), 2)
        
        # Draw vertical line on the right side of the frame
        cv2.line(frame, (width - 50, 0), (width - 50, height), (255, 0, 255), 2)

        # Calculate and display FPS
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time

        # Display FPS on the screen
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Print the value of "part depan"
        cv2.putText(frame, f"Part Depan Value: {part_depan_value}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Part Belakang Value: {part_belakang_value}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("YOLO Object Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_objects()
