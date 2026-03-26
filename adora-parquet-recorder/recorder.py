import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from adora import Node

LOG_DIR = os.getenv("LOG_DIR", "logs")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 30))

os.makedirs(LOG_DIR, exist_ok=True)

node = Node()

# 发送就绪信号，通知 camera 节点可以开始
node.send_output("status", pa.array(["ready"]))
print(f"Recorder ready. batch_size={BATCH_SIZE}, log_dir={LOG_DIR}", flush=True)

batch = []
file_index = 0


def flush_batch(batch, file_index):
    if not batch:
        return file_index
    df = pd.DataFrame(batch)
    path = os.path.join(LOG_DIR, f"frames_{file_index:04d}.parquet")
    df.to_parquet(path, index=False)
    print(f"Saved {len(batch)} frames → {path}", flush=True)
    return file_index + 1


for event in node:
    if event["type"] == "INPUT" and event["id"] == "image":
        metadata = event["metadata"]
        frame_data = event["value"].to_numpy().tobytes()
        batch.append(
            {
                "timestamp_ns": time.perf_counter_ns(),
                "frame_id": metadata.get("frame_id", len(batch)),
                "width": metadata.get("width", 640),
                "height": metadata.get("height", 480),
                "encoding": metadata.get("encoding", "bgr8"),
                "data": frame_data,
            }
        )
        if len(batch) >= BATCH_SIZE:
            file_index = flush_batch(batch, file_index)
            batch = []
    elif event["type"] == "STOP":
        break

# 写入剩余帧
flush_batch(batch, file_index)
print("Recorder finished.", flush=True)
