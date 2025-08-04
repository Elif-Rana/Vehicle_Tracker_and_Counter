import supervision as sv
import numpy as np
import time
import os

from utils.line_utils import analyze_vehicle_flow, create_multiple_counting_lines
from utils.line_crossing_tracker import LineCrossingTracker
from utils.roi_utils import crop_detections_to_roi, draw_roi_on_frame

def count_vehicles_multiline(model, vehicle_ids, 
                             input_video_path, output_video_path, stats_output_path,
                             roi_coords, frame_skip=0,
                             num_lines=3) -> int:
    """
    Count vehicles crossing multiple counting lines with tracking.
    
    Args:
        model: YOLO model instance for vehicle detection
        vehicle_ids: List of class IDs for vehicle types to detect
        input_video_path (str): Path to input video file
        output_video_path (str): Path to save annotated output video
        stats_output_path (str): Path to save counting statistics
        num_lines (int): Number of counting lines to create. Defaults to 3.
    
    Returns:
        int: Total number of unique vehicles that crossed any line
    """

    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stats_dir = os.path.dirname(stats_output_path)
    if stats_dir and not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
        
    frame_times = []

    generator = sv.get_video_frames_generator(input_video_path)

    frame = next(iter(generator))
    height, width, _ = frame.shape

    # Adaptive sizing based on video dimensions for aesthetic appearance
    base_size = min(width, height)
    line_thickness = max(2, int(base_size / 400))
    line_text_thickness = max(1, int(base_size / 600))
    line_text_scale = max(0.5, base_size / 1200)
    box_thickness = max(1, int(base_size / 500))
    label_text_thickness = max(1, int(base_size / 800))
    label_text_scale = max(0.4, base_size / 1500)

    # vehicle flow patterns
    line_start, line_end, direction = analyze_vehicle_flow(model=model, vehicle_ids=vehicle_ids, input_video_path=input_video_path)

    if roi_coords:
        x_min, y_min, x_max, y_max = roi_coords
        roi_width = x_max - x_min
        roi_height = y_max - y_min
        print(f"Using ROI: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        
        # Create counting lines within ROI
        counting_lines = create_multiple_counting_lines(roi_width, roi_height, direction, num_lines)
        # Adjust line coordinates to ROI offset
        for line in counting_lines:
            line.vector.start.x += x_min
            line.vector.start.y += y_min
            line.vector.end.x += x_min
            line.vector.end.y += y_min
    else:
        roi_width, roi_height = width, height
        counting_lines = create_multiple_counting_lines(width, height, direction, num_lines)
        print("Processing entire frame")

    crossing_tracker = LineCrossingTracker(num_lines)

    # Annotators for each line
    line_annotators = []
    for i, line in enumerate(counting_lines):
        if direction == "horizontal":
            custom_in = f"Line {i+1}: Bottom to Top"
            custom_out = f"Line {i+1}: Top to Bottom"
        else:
            custom_in = f"Line {i+1}: Left to Right"
            custom_out = f"Line {i+1}: Right to Left"

        annotator = sv.LineZoneAnnotator(
            thickness=line_thickness,
            text_thickness=line_text_thickness,
            text_scale=line_text_scale,
            custom_in_text=custom_in,
            custom_out_text=custom_out
        )
        line_annotators.append(annotator)

    # Tracker
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
        minimum_consecutive_frames=3
    )

    # Reset generator for main processing
    generator = sv.get_video_frames_generator(input_video_path)

    # Annotators with adaptive sizing
    box_annotator = sv.BoxAnnotator(thickness=box_thickness)
    label_annotator = sv.LabelAnnotator(
        text_thickness=label_text_thickness,
        text_scale=label_text_scale,
        text_color=sv.Color.BLACK
    )

    frame_skip = frame_skip # If other than 0, then process the frames by skipping
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        """
        Process each video frame for vehicle detection and counting.
        
        Args:
            frame (np.ndarray): Current video frame
            index (int): Frame index in the video sequence
        
        Returns:
            np.ndarray: Annotated frame with detections, tracking, and counting lines
        """
        
        if index % (frame_skip+1) != 0:
            return frame
        start_time = time.time()

        if roi_coords:
            x_min, y_min, x_max, y_max = roi_coords
            # Crop frame to ROI before running model
            cropped_frame = frame[y_min:y_max, x_min:x_max]
            results = model(cropped_frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Adjust detection coordinates back to full frame space
            if len(detections) > 0:
                detections.xyxy[:, 0] += x_min  # x1
                detections.xyxy[:, 1] += y_min  # y1
                detections.xyxy[:, 2] += x_min  # x2
                detections.xyxy[:, 3] += y_min  # y2
        else:
            # Process full frame
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
        
        # Filter for vehicle classes
        detections = detections[np.isin(detections.class_id, vehicle_ids)]
        
        # Update tracker with detections
        detections = byte_tracker.update_with_detections(detections)
        
        # Update crossing tracker - THIS WAS THE MISSING PIECE!
        crossing_tracker.update_vehicle_positions(
            detections, counting_lines, direction, model.model.names
        )
        
        # Create labels for visualization
        labels = [
            f"#{model.model.names[class_id].capitalize()} {tracker_id}" 
            for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
            if tracker_id is not None
        ]
        
        # Annotate frame
        annotated_frame = frame.copy()
        
        # Draw ROI boundary if using ROI
        if roi_coords:
            annotated_frame = draw_roi_on_frame(annotated_frame, roi_coords)
        
        # Draw detection boxes and labels
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        # Update line counters for visualization (separate from our custom tracker)
        for line in counting_lines:
            line.trigger(detections)
        
        # Draw counting lines with their counters
        for line, annotator in zip(counting_lines, line_annotators):
            annotated_frame = annotator.annotate(annotated_frame, line_counter=line)
        
        # Record processing time
        end_time = time.time()
        frame_times.append((end_time - start_time) * 1000)
        
        return annotated_frame

    # Process video
    sv.process_video(
        source_path=input_video_path,
        target_path=output_video_path,
        callback=callback
    )

    # CALCULATE STATS
    unique_vehicle_count = crossing_tracker.get_unique_vehicle_count()

    # Calculate totals from tracker (not the line counters)
    total_in = sum(crossing_tracker.get_line_stats(i)["in_count"] for i in range(num_lines))
    total_out = sum(crossing_tracker.get_line_stats(i)["out_count"] for i in range(num_lines))

    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0

    total_frames = len(frame_times) * (frame_skip + 1) 
    total_processing_time = sum(frame_times)
    avg_effective_time_per_frame = total_processing_time / total_frames if total_frames > 0 else 0

    with open(stats_output_path, "w") as f:
        if roi_coords:
            f.write(f"ROI coordinates: ({roi_coords[0]}, {roi_coords[1]}) to ({roi_coords[2]}, {roi_coords[3]})\n")
        f.write(f"Number of counting lines: {num_lines}\n")
        f.write(f"Total unique vehicles that crossed lines: {unique_vehicle_count}\n")
        f.write(f"Total crossings: IN={total_in}, OUT={total_out}\n")
        f.write(f"Average frame processing time: {avg_frame_time:.2f} ms\n")
        f.write(f"Number of frames skipped: {frame_skip}\n")
        f.write(f"Average effective time per video frame: {avg_effective_time_per_frame:.2f} ms\n")


        for i in range(num_lines):
            line_stats = crossing_tracker.get_line_stats(i)
            f.write(f"\nLine {i+1}:\n")
            f.write(f"  Total: IN={line_stats['in_count']}, OUT={line_stats['out_count']}\n")

            if line_stats['in_vehicles']:
                f.write(f"  Vehicles going IN: ")
                vehicle_list = [f"{class_name} #{tracker_id}" for tracker_id, class_name in line_stats['in_vehicles']]
                f.write(", ".join(vehicle_list) + "\n")

            if line_stats['out_vehicles']:
                f.write(f"  Vehicles going OUT: ")
                vehicle_list = [f"{class_name} #{tracker_id}" for tracker_id, class_name in line_stats['out_vehicles']]
                f.write(", ".join(vehicle_list) + "\n")

    return unique_vehicle_count