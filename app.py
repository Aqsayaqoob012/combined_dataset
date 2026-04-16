import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# =========================
# LOAD MODEL
# =========================
model = YOLO("best (1).pt")

st.title("🚦 Smart Traffic AI System")

video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

# =========================
# CLASS MAP
# =========================
class_names = {
    0: "car",
    1: "bike",
    2: "rickshaw",
    3: "bus",
    4: "truck",
    5: "van",
    6: "cart",
    7: "helmet",
    8: "no_helmet"
}

vehicle_classes = ["car","bike","rickshaw","bus","truck","van","cart"]

# =========================
# COUNTERS
# =========================
unique_ids = set()
class_count = defaultdict(int)

# =========================
# TRACK MEMORY (FIX DUPLICATE)
# =========================
memory = {}

def is_new_object(obj_id, cx, cy):
    """
    Fix duplicate counting:
    same object again detect na ho
    """
    if obj_id in memory:
        px, py = memory[obj_id]
        dist = ((cx - px)**2 + (cy - py)**2) ** 0.5

        if dist < 60:   # same vehicle threshold
            memory[obj_id] = (cx, cy)
            return False

    memory[obj_id] = (cx, cy)
    return True

# =========================
# PROCESS VIDEO
# =========================
if video_file:

    with open("video.mp4", "wb") as f:
        f.write(video_file.read())

    frame_slot = st.empty()
    stats_slot = st.empty()

    # =========================
    # TRACKING MODE
    # =========================
    results = model.track(
        source="video.mp4",
        stream=True,
        persist=True,
        tracker="bytetrack.yaml"
    )

    for r in results:

        frame = r.orig_img
        annotated = frame.copy()

        if r.boxes.id is not None:

            for box, cls, obj_id in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.id):

                cls = int(cls)
                obj_id = int(obj_id)

                x1, y1, x2, y2 = map(int, box)

                label = class_names.get(cls, "unknown")

                # center point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # =========================
                # DRAW BOX (STABLE)
                # =========================
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)

                cv2.putText(
                    annotated,
                    f"{label} ID:{obj_id}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

                # =========================
                # UNIQUE VEHICLE COUNT FIX
                # =========================
                if label in vehicle_classes:

                    if is_new_object(obj_id, cx, cy):

                        unique_ids.add(obj_id)
                        class_count[label] += 1

                # =========================
                # HELMET ALERT
                # =========================
                if label == "no_helmet":
                    cv2.putText(
                        annotated,
                        "NO HELMET ⚠",
                        (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,255),
                        2
                    )

        # =========================
        # DASHBOARD
        # =========================
        total = len(unique_ids)

        stats = f"""
🚗 TOTAL VEHICLES: {total}

🚙 Car: {class_count['car']}
🏍 Bike: {class_count['bike']}
🛺 Rickshaw: {class_count['rickshaw']}
🚌 Bus: {class_count['bus']}
🚚 Truck: {class_count['truck']}
🚐 Van: {class_count['van']}
🛒 Cart: {class_count['cart']}
        """

        frame_slot.image(annotated, channels="BGR")
        stats_slot.markdown(stats)