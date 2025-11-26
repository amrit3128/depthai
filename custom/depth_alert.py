import depthai as dai
import numpy as np
import cv2
import platform
import time

# Function to play beep sound on different OS
def beep():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 200)
    else:
        # Linux/macOS (bell sound in terminal)
        print('\a')

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

monoLeft = pipeline.createMonoCamera()
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

monoRight = pipeline.createMonoCamera()
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputDepth(True)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

nn = pipeline.createMobileNetSpatialDetectionNetwork()
# Use the built-in model "mobilenet-ssd" which automatically downloads the blob
nn.setBlobPath(dai.ModelPath.MOBILENET_DETECTION)
nn.setConfidenceThreshold(0.5)
nn.input.setBlocking(False)
camRgb.preview.link(nn.input)

xoutNN = pipeline.createXLinkOut()
xoutNN.setStreamName("detections")
nn.out.link(xoutNN.input)

xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    THRESHOLD_DISTANCE = 1000  # mm distance to trigger beep

    while True:
        inDet = qDet.tryGet()
        inRgb = qRgb.tryGet()
        inDepth = qDepth.tryGet()

        if inRgb is None or inDepth is None:
            continue

        frame = inRgb.getCvFrame()
        depthFrame = inDepth.getFrame()

        if inDet is not None:
            for detection in inDet.detections:
                if detection.label == 15:  # Person class for MobileNetSSD COCO
                    x1 = int(detection.xmin * frame.shape[1])
                    y1 = int(detection.ymin * frame.shape[0])
                    x2 = int(detection.xmax * frame.shape[1])
                    y2 = int(detection.ymax * frame.shape[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 255, 10), 2)

                    # depth at bbox center
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    depth = depthFrame[cy, cx]

                    cv2.putText(frame, f"Depth: {depth}mm", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 10), 2)

                    if 0 < depth < THRESHOLD_DISTANCE:
                        print("Person too close! Beep!")
                        beep()

        cv2.imshow("Person Detection with Proximity", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
