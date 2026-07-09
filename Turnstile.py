from ultralytics import YOLO
import cv2
import cvzone
import datetime
import sqlite3
from threading import Thread, Lock  # Added explicit Lock primitive
import time


# =====================================================================
# STEP 1: MULTI-THREADED THREAD-SAFE VIDEO CAPTURE DEFINITION
# =====================================================================
class RealTimeVideoStream:
    """Reads video frames inside a dedicated background thread to prevent pipeline stalls.
    Utilizes a Mutex Lock to ensure mutual exclusion across shared resources."""

    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = Lock()  # Instantiate the Mutex Lock

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
                return

            # Read into temporary local variables first outside the lock
            # to minimize the time spent holding the critical section.
            grabbed, frame = self.stream.read()

            # Acquire the lock before updating the shared class state
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        # Acquire lock before reading shared memory to prevent frame tearing/stale state reads
        with self.lock:
            return self.frame if self.grabbed else None

    def stop(self):
        self.stopped = True
        self.stream.release()


# =====================================================================
# STEP 2: SQL DATABASE SETUP & SYSTEM OPTIMIZATION
# =====================================================================
db_conn = sqlite3.connect("turnstile_security.db")
db_cursor = db_conn.cursor()

# Wipe out the old table structure completely to clear any stale states
db_cursor.execute("DROP TABLE IF EXISTS security_logs")
db_conn.commit()

# Re-create a completely fresh, empty table structure
db_cursor.execute('''
    CREATE TABLE IF NOT EXISTS security_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER,
        direction TEXT,
        timestamp TEXT
    )
''')
db_conn.commit()

# Creates a B-Tree index on the timestamp column for optimized query performance
db_cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_security_timestamp 
    ON security_logs (timestamp)
''')
db_conn.commit()
print("[DATABASE RESET] Old tables dropped. Fresh database with query indexing initialized.")


def log_event_to_sql(p_id, movement_dir):
    """Executes structural query entry into local SQL table storage."""
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db_cursor.execute(
        "INSERT INTO security_logs (person_id, direction, timestamp) VALUES (?, ?, ?)",
        (int(p_id), movement_dir, timestamp_str)
    )
    db_conn.commit()
    print(f"[SQL INSERT SUCCESS] Person #{p_id} registered as: '{movement_dir}' at {timestamp_str}")


# =====================================================================
# STEP 3: PIPELINE INITIALIZATION
# =====================================================================
# Instantiate and fire up the background frame-grabbing thread
video_stream = RealTimeVideoStream("../Videos/turnstile.mp4").start()
time.sleep(1.0)  # Safe buffer allowing the sensor/file stream decode to warm up

model = YOLO("../YoloWeights/yolov8m.pt").to('cuda')

# Define full coordinates for the angled lines: (pt1_x, pt1_y), (pt2_x, pt2_y)
LEFT_LINE = ((550, 1330), (1320, 1410))
RIGHT_LINE = ((1930, 1370), (3100, 1480))


def get_line_y_at_x(target_line, current_x):
    """Calculates the expected Y position on a slanted line using y = mx + b"""
    (line_x1, line_y1), (line_x2, line_y2) = target_line
    if line_x2 - line_x1 == 0:
        return line_y1
    slope = (line_y2 - line_y1) / (line_x2 - line_x1)
    intercept = line_y1 - (slope * line_x1)
    return int((slope * current_x) + intercept)


previous_positions = {}
already_logged_ids = set()

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1280, 720)

# =====================================================================
# STEP 4: CORE EXECUTION FRAME LOOP
# =====================================================================
while True:
    # Safely fetch the thread-safe frame pre-loaded via the background thread
    img = video_stream.read()
    if img is None:
        print("End of stream or failed to grab frame. Exiting...")
        break

    results = model.track(img, persist=True, stream=True, verbose=False, classes=[0], tracker="bytetrack.yaml")

    # Draw Angled Lines
    cv2.line(img, LEFT_LINE[0], LEFT_LINE[1], (0, 0, 255), 3)
    cv2.putText(img, "LEFT BOUNDARY (ENTER ONLY)", (LEFT_LINE[0][0], LEFT_LINE[0][1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.line(img, RIGHT_LINE[0], RIGHT_LINE[1], (255, 255, 0), 3)
    cv2.putText(img, "RIGHT BOUNDARY (LEAVE ONLY)", (RIGHT_LINE[0][0] + 20, RIGHT_LINE[0][1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    for r in results:
        if r.boxes.id is not None:
            boxes_array = r.boxes.xyxy.cpu().numpy()
            clss_array = r.boxes.cls.cpu().numpy()
            confs_array = r.boxes.conf.cpu().numpy()
            track_ids_array = r.boxes.id.cpu().numpy()

            for single_box, single_cls, single_conf, single_id in zip(boxes_array, clss_array, confs_array,
                                                                      track_ids_array):
                box_x1, box_y1, box_x2, box_y2 = map(int, single_box)
                box_w, box_h = box_x2 - box_x1, box_y2 - box_y1

                # Calculate tracking target coordinates (Lower leg shin area)
                cx_point = int((box_x1 + box_x2) / 2)
                cy_point = int(box_y2 - (box_h * 0.08))

                # Visuals
                cvzone.cornerRect(img, (box_x1, box_y1, box_w, box_h), l=20, t=3, rt=0, colorR=(255, 0, 0))
                cv2.circle(img, (cx_point, cy_point), 6, (0, 255, 0), -1)

                conf_val = round(float(single_conf), 2)
                cvzone.putTextRect(img, f'Person {int(single_id)} {conf_val}', (max(0, box_x1), max(45, box_y1)),
                                   scale=1.2, thickness=2, offset=5, colorT=(255, 255, 255), colorR=(255, 0, 0))

                # -----------------------------------------------------
                # CROSSING INTERSECTION LOGIC (FILTERED DIRECTION)
                # -----------------------------------------------------
                if single_id in previous_positions:
                    past_cy = previous_positions[single_id]

                    if single_id not in already_logged_ids:

                        # Split left vs right side using screen center
                        if cx_point < (img.shape[1] // 2):
                            target_line_y = get_line_y_at_x(LEFT_LINE, cx_point)

                            # ENTERING ONLY: Downward crossing check
                            if past_cy < target_line_y <= cy_point:
                                log_event_to_sql(single_id, "Entering (Left)")
                                already_logged_ids.add(single_id)

                        # Right side of the screen context
                        else:
                            target_line_y = get_line_y_at_x(RIGHT_LINE, cx_point)

                            # LEAVING ONLY: Upward crossing check
                            if past_cy > target_line_y >= cy_point:
                                log_event_to_sql(single_id, "Leaving (Right)")
                                already_logged_ids.add(single_id)

                previous_positions[single_id] = cy_point

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean cleanup
video_stream.stop()
cv2.destroyAllWindows()
db_conn.close()