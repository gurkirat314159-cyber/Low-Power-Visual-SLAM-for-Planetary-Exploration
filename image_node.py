#!/usr/bin/env python3
"""
image_node.py  -  rover_pipeline

Subscribes to a camera image, applies CLAHE (to fight cave shadows/glare),
detects FAST corners, tracks them frame to frame with pyramidal KLT
(with a forward-backward check), re-detects when too few remain, and
publishes an annotated image you can view in RViz2 / rqt_image_view.

Parameters (override with --ros-args -p name:=value):
  input_topic   (str)   /image_raw
  output_topic  (str)   /rover/image_enhanced
  clahe_clip    (float) 3.0
  clahe_grid    (int)   8
  fast_thresh   (int)   20
  max_corners   (int)   300
  min_corners   (int)   120     re-detect below this many tracked points
  min_distance  (int)   12      min pixel spacing between corners
  fb_max_error  (float) 1.0     forward-backward error limit in pixels
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')

        self.declare_parameter('input_topic', '/image_raw')
        self.declare_parameter('output_topic', '/rover/image_enhanced')
        self.declare_parameter('clahe_clip', 3.0)
        self.declare_parameter('clahe_grid', 8)
        self.declare_parameter('fast_thresh', 20)
        self.declare_parameter('max_corners', 300)
        self.declare_parameter('min_corners', 120)
        self.declare_parameter('min_distance', 12)
        self.declare_parameter('fb_max_error', 1.0)

        p = self.get_parameter
        in_topic = p('input_topic').value
        out_topic = p('output_topic').value
        self.max_corners = int(p('max_corners').value)
        self.min_corners = int(p('min_corners').value)
        self.min_distance = int(p('min_distance').value)
        self.fb_max_error = float(p('fb_max_error').value)

        grid = int(p('clahe_grid').value)
        # Create once, reuse every frame.
        self.clahe = cv2.createCLAHE(
            clipLimit=float(p('clahe_clip').value), tileGridSize=(grid, grid))
        self.fast = cv2.FastFeatureDetector_create(
            threshold=int(p('fast_thresh').value), nonmaxSuppression=True)

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.bridge = CvBridge()
        self.prev_gray = None
        self.prev_pts = np.empty((0, 1, 2), dtype=np.float32)

        # Sensor QoS (BEST_EFFORT) matches most camera drivers.
        self.sub = self.create_subscription(
            Image, in_topic, self.image_callback, qos_profile_sensor_data)
        self.pub = self.create_publisher(
            Image, out_topic, qos_profile_sensor_data)

        self.get_logger().info(f'Listening on {in_topic}, publishing {out_topic}')

    # ------------------------------------------------------------------ #
    def detect_new(self, gray, existing_pts, need):
        """Detect up to `need` FAST corners, avoiding existing points."""
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for x, y in existing_pts.reshape(-1, 2):
            cv2.circle(mask, (int(x), int(y)), self.min_distance, 0, -1)

        kps = self.fast.detect(gray, mask)
        kps = sorted(kps, key=lambda k: -k.response)

        chosen = []
        taken = np.zeros(gray.shape, dtype=np.uint8)  # enforce spacing among new ones
        for k in kps:
            if len(chosen) >= need:
                break
            x, y = int(k.pt[0]), int(k.pt[1])
            if taken[y, x]:
                continue
            chosen.append(k.pt)
            cv2.circle(taken, (x, y), self.min_distance, 255, -1)

        if not chosen:
            return np.empty((0, 1, 2), dtype=np.float32)
        return np.array(chosen, dtype=np.float32).reshape(-1, 1, 2)

    # ------------------------------------------------------------------ #
    def track(self, gray):
        """KLT track prev_pts into `gray` with forward-backward validation."""
        if self.prev_gray is None or len(self.prev_pts) == 0:
            return np.empty((0, 1, 2), dtype=np.float32)

        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params)
        back, st_b, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, nxt, None, **self.lk_params)

        fb_err = np.abs(self.prev_pts - back).reshape(-1, 2).max(axis=1)
        h, w = gray.shape
        xy = nxt.reshape(-1, 2)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)

        good = (st.ravel() == 1) & (st_b.ravel() == 1) & \
               (fb_err < self.fb_max_error) & inside
        return nxt[good].reshape(-1, 1, 2).astype(np.float32)

    # ------------------------------------------------------------------ #
    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)

        pts = self.track(enhanced)
        tracked = len(pts)

        if tracked < self.min_corners:
            new_pts = self.detect_new(
                enhanced, pts, self.max_corners - tracked)
            if len(new_pts):
                pts = np.concatenate([pts, new_pts], axis=0)

        # Annotate: enhanced image in colour, tracked points in green.
        out = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        for x, y in pts.reshape(-1, 2):
            cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.putText(out, f'tracked: {tracked}  total: {len(pts)}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out_msg = self.bridge.cv2_to_imgmsg(out, encoding='bgr8', header=msg.header)
        self.pub.publish(out_msg)

        self.prev_gray = enhanced
        self.prev_pts = pts.astype(np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
