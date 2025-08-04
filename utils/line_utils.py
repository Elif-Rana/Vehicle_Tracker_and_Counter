from collections import defaultdict
from typing import Tuple, List
import supervision as sv
import numpy as np

def analyze_vehicle_flow(model, vehicle_ids,
                         input_video_path: str, sample_frames: int = 200) -> Tuple[sv.Point, sv.Point, str]:
    """
    Analyze vehicle movement pattern to determine optimal counting line placement.
    
    Args:
        model: YOLO model instance for vehicle detection
        vehicle_ids: List of class IDs for vehicle types to detect
        input_video_path (str): Path to input video file
        sample_frames (int): Number of frames to analyze. Defaults to 100.
    
    Returns:
        Tuple[sv.Point, sv.Point, str]: Start point, end point, and orientation 
        ("horizontal" or "vertical") of the optimal counting line.
    """
    generator = sv.get_video_frames_generator(input_video_path)

    frame = next(iter(generator))
    height, width, _ = frame.shape

    # Tracking for traffic flow analysis
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
        minimum_consecutive_frames=3
    )

    trajectories = defaultdict(list)
    frame_count = 0

    for frame in generator:
        if frame_count >= sample_frames:
            break

        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, vehicle_ids)]

        detections = byte_tracker.update_with_detections(detections)

        # Use centroids to track -> less affected by minor variations than bbox
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is not None:
                bbox = detections.xyxy[i]
                centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                trajectories[tracker_id].append(centroid)

        frame_count += 1

    horizontal_movements = []
    vertical_movements = []

    for track_id, points in trajectories.items():
        if len(points) >= 5:  # Minimum points for reliable direction (Gave it 5 hard coded)
            start_point = points[0]
            end_point = points[-1]

            dx = end_point[0] - start_point[0]  # horizontal displacement
            dy = end_point[1] - start_point[1]  # vertical displacement

            horizontal_movements.append(abs(dx))
            vertical_movements.append(abs(dy))

    # If the traffic flow cannot be found in given frames (200), then our default is a horizontal line in the center
    # (This decision was made by considering the test video "highway.ts")
    if not horizontal_movements and not vertical_movements:
        return (sv.Point(0, height // 2), sv.Point(width, height // 2), "horizontal")   # center horizontal line

    avg_horizontal = np.mean(horizontal_movements) if horizontal_movements else 0
    avg_vertical = np.mean(vertical_movements) if vertical_movements else 0

    if avg_horizontal > avg_vertical:   # Dominant horizontal movement -> vertical counting line
        x_pos = int(width * 0.5)
        return (sv.Point(x_pos, 0), sv.Point(x_pos, height), "vertical")
    else:
        y_pos = int(height * 0.5)   # Dominant vertical movement -> horizontal counting line
        return (sv.Point(0, y_pos), sv.Point(width, y_pos), "horizontal")

def create_multiple_counting_lines(width: int, height: int, direction: str, num_lines: int = 3) -> List[sv.LineZone]:
    """
    Create multiple counting lines that divide the video into equal segments.
    
    Args:
        width (int): Video frame width in pixels
        height (int): Video frame height in pixels
        direction (str): Line orientation, either "horizontal" or "vertical"
        num_lines (int): Number of counting lines to create. Defaults to 3.
    
    Returns:
        List[sv.LineZone]: List of LineZone objects representing the counting lines.
        Creates (num_lines + 1) equal segments across the frame.
    """
    lines = []
    # For n lines -> (n+1) equal segments
    # Single line -> at center
    if num_lines == 1:

        if direction == "horizontal":
            y_pos = int(height * 0.5)
            lines.append(sv.LineZone(start=sv.Point(0, y_pos), end=sv.Point(width, y_pos)))
        else:
            x_pos = int(width * 0.5)
            lines.append(sv.LineZone(start=sv.Point(x_pos, 0), end=sv.Point(x_pos, height)))

    # num_lines lines that equally far away from each other
    else:

        if direction == "horizontal":            
            segment_height = height / (num_lines + 1)
            for i in range(num_lines):   
                y_pos = int(segment_height * (i + 1))
                lines.append(sv.LineZone(start=sv.Point(0, y_pos), end=sv.Point(width, y_pos)))
        else:
            segment_width = width / (num_lines + 1)
            for i in range(num_lines):
                x_pos = int(segment_width * (i + 1))
                lines.append(sv.LineZone(start=sv.Point(x_pos, 0), end=sv.Point(x_pos, height)))

    return lines