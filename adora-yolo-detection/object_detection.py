import numpy as np
import pyarrow as pa
from adora import Node
from ultralytics import YOLO

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# 强制 CPU，避免多进程抢 GPU 显存导致 SIGSEGV
model = YOLO("yolov8n.pt")
model.to("cpu")


def main():
    node = Node()
    for event in node:
        if event["type"] == "INPUT":
            raw = event["value"].to_numpy()

            # 校验帧大小，防止残帧导致 reshape 崩溃
            expected = CAMERA_HEIGHT * CAMERA_WIDTH * 3
            if raw.size != expected:
                continue

            frame = raw.reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 3))
            frame = frame[:, :, ::-1]  # BGR to RGB

            results = model(frame, verbose=False, device="cpu")

            boxes = np.array(results[0].boxes.xyxy.cpu())
            conf  = np.array(results[0].boxes.conf.cpu())
            label = np.array(results[0].boxes.cls.cpu())

            if len(boxes) > 0:
                arrays = np.concatenate((boxes, conf[:, None], label[:, None]), axis=1)
            else:
                arrays = np.empty((0, 6), dtype=np.float32)

            node.send_output("bbox", pa.array(arrays.ravel()), event["metadata"])

        elif event["type"] == "STOP":
            break


if __name__ == "__main__":
    main()
