import cv2
import numpy as np
from typing import Tuple, Optional
import supervision as sv


class InteractiveROISelector:
    """
    Interactive ROI selection tool for video frames.
    """
    
    def __init__(self):
        """Initialize the ROI selector."""
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.roi_coords = None
        self.original_frame = None
        self.display_frame = None
        
    def _calculate_display_params(self, frame_shape):
        """
        Calculate display parameters based on frame dimensions.
        
        Args:
            frame_shape: Frame shape (height, width)
        
        Returns:
            tuple: (line_thickness, font_scale, text_thickness)
        """
        height, width = frame_shape[:2]
        
        # Base calculations on frame resolution
        base_size = min(width, height)
        
        # Scale parameters based on resolution
        if base_size <= 480: 
            line_thickness = 2
            font_scale = 0.8
            text_thickness = 2
        elif base_size <= 720: 
            line_thickness = 3
            font_scale = 1.2
            text_thickness = 2
        elif base_size <= 1080: 
            line_thickness = 4
            font_scale = 1.5
            text_thickness = 3
        elif base_size <= 1440: 
            line_thickness = 5
            font_scale = 2.0
            text_thickness = 4
        else: 
            line_thickness = 6
            font_scale = 2.5
            text_thickness = 5
        
        return line_thickness, font_scale, text_thickness
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events for ROI selection.
        
        Args:
            event: OpenCV mouse event type
            x (int): Mouse x coordinate
            y (int): Mouse y coordinate
            flags: Additional event flags
            param: Additional parameters
        """
        line_thickness, font_scale, text_thickness = self._calculate_display_params(self.original_frame.shape)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
            self.display_frame = self.original_frame.copy()
            cv2.rectangle(self.display_frame, self.start_point, self.end_point, (0, 255, 0), line_thickness)
            cv2.imshow('Select ROI - Press SPACE to confirm, ESC to cancel', self.display_frame)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            self.roi_coords = (
                min(x1, x2),  # x_min
                min(y1, y2),  # y_min
                max(x1, x2),  # x_max
                max(y1, y2)   # y_max
            )
            
            # Final rectangle with scaled parameters
            self.display_frame = self.original_frame.copy()
            cv2.rectangle(self.display_frame, (self.roi_coords[0], self.roi_coords[1]), 
                         (self.roi_coords[2], self.roi_coords[3]), (0, 255, 0), line_thickness)
            
            # Calculate text positioning based on frame size
            height, width = self.original_frame.shape[:2]
            text_y = int(height * 0.08) if height > 200 else 40
            
            cv2.putText(self.display_frame, 'Press SPACE to confirm, ESC to cancel', 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thickness)
            cv2.imshow('Select ROI - Press SPACE to confirm, ESC to cancel', self.display_frame)


def select_roi_from_video(video_path: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Select ROI from a video frame interactively.
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        Optional[Tuple[int, int, int, int]]: ROI coordinates (x_min, y_min, x_max, y_max) 
        or None if cancelled
    """
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    # ROI selector
    roi_selector = InteractiveROISelector()
    roi_selector.original_frame = frame.copy()
    roi_selector.display_frame = frame.copy()
    
    # Create window and set mouse callback
    cv2.namedWindow('Select ROI - Press SPACE to confirm, ESC to cancel', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select ROI - Press SPACE to confirm, ESC to cancel', roi_selector.mouse_callback)
    
    # INSTRUCTIONS
    instructions = frame.copy()
    height, width = frame.shape[:2]
    line_thickness, font_scale, text_thickness = InteractiveROISelector()._calculate_display_params(frame.shape)
    
    # text positioning based on frame size
    text_y1 = int(height * 0.08) if height > 200 else 40
    text_y2 = int(height * 0.15) if height > 200 else 80
    
    cv2.putText(instructions, 'Click and drag to select ROI', (10, text_y1), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thickness)
    cv2.putText(instructions, 'Press SPACE to confirm, ESC to cancel', (10, text_y2), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thickness)
    cv2.imshow('Select ROI - Press SPACE to confirm, ESC to cancel', instructions)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # SPACE key
            if roi_selector.roi_coords is not None:
                cv2.destroyAllWindows()
                return roi_selector.roi_coords
            else:
                print("Please select an ROI first")
                
        elif key == 27:  # ESC key
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    return None


def crop_detections_to_roi(detections: sv.Detections, roi_coords: Tuple[int, int, int, int]) -> sv.Detections:
    """
    Filter detections to only include those within the ROI.
    
    Args:
        detections (sv.Detections): Original detections
        roi_coords (Tuple[int, int, int, int]): ROI coordinates (x_min, y_min, x_max, y_max)
    
    Returns:
        sv.Detections: Filtered detections within ROI
    """
    if len(detections) == 0:
        return detections
    
    x_min, y_min, x_max, y_max = roi_coords
    
    # detection centroids
    centroids_x = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2
    centroids_y = (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2
    
    # Filter detections whose centroids are within ROI
    mask = (
        (centroids_x >= x_min) & (centroids_x <= x_max) &
        (centroids_y >= y_min) & (centroids_y <= y_max)
    )
    
    return detections[mask]


def draw_roi_on_frame(frame: np.ndarray, roi_coords: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Draw ROI rectangle on frame.
    
    Args:
        frame (np.ndarray): Input frame
        roi_coords (Tuple[int, int, int, int]): ROI coordinates (x_min, y_min, x_max, y_max)
    
    Returns:
        np.ndarray: Frame with ROI rectangle drawn
    """
    x_min, y_min, x_max, y_max = roi_coords
    height, width = frame.shape[:2]
    
    # adaptive size parameters
    base_size = min(width, height)
    
    if base_size <= 480:
        line_thickness = 1
        font_scale = 0.4
        text_thickness = 1
    elif base_size <= 720:
        line_thickness = 2
        font_scale = 0.6
        text_thickness = 1
    elif base_size <= 1080:
        line_thickness = 2
        font_scale = 0.7
        text_thickness = 2
    elif base_size <= 1440:
        line_thickness = 3
        font_scale = 0.9
        text_thickness = 2
    else:
        line_thickness = 4
        font_scale = 1.2
        text_thickness = 3
    
    # Draw rectangle
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), line_thickness)
    # text posiiton
    text_y = y_min - 20 if y_min > 50 else y_min + 40
    
    cv2.putText(frame, 'ROI', (x_min, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), text_thickness)
    return frame


def extract_roi_from_frame(frame: np.ndarray, roi_coords: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Extract ROI region from frame for processing.
    
    Args:
        frame (np.ndarray): Input frame
        roi_coords (Tuple[int, int, int, int]): ROI coordinates (x_min, y_min, x_max, y_max)
    
    Returns:
        np.ndarray: Cropped frame containing only ROI
    """
    x_min, y_min, x_max, y_max = roi_coords
    return frame[y_min:y_max, x_min:x_max]


def adjust_detections_to_full_frame(detections: sv.Detections, roi_coords: Tuple[int, int, int, int]) -> sv.Detections:
    """
    Adjust detection coordinates from ROI space back to full frame space.
    
    Args:
        detections (sv.Detections): Detections in ROI coordinate space
        roi_coords (Tuple[int, int, int, int]): ROI coordinates (x_min, y_min, x_max, y_max)
    
    Returns:
        sv.Detections: Detections adjusted to full frame coordinates
    """
    if len(detections) == 0:
        return detections
    
    x_min, y_min, _, _ = roi_coords
    adjusted_detections = detections.copy()
    
    # Adjust bounding box coordinates
    adjusted_detections.xyxy[:, 0] += x_min  # x1
    adjusted_detections.xyxy[:, 1] += y_min  # y1
    adjusted_detections.xyxy[:, 2] += x_min  # x2
    adjusted_detections.xyxy[:, 3] += y_min  # y2
    
    return adjusted_detections


def create_roi_processor(roi_coords: Tuple[int, int, int, int]):
    """
    Create a processor function that handles ROI cropping and coordinate adjustment.
    
    Args:
        roi_coords (Tuple[int, int, int, int]): ROI coordinates (x_min, y_min, x_max, y_max)
    
    Returns:
        Function that processes frames with ROI optimization
    """
    def process_frame_with_roi(frame: np.ndarray, detection_function) -> sv.Detections:
        """
        Process frame with ROI.
        
        Args:
            frame (np.ndarray): Full frame
            detection_function: Function that performs detection on cropped frame
        
        Returns:
            sv.Detections: Detections adjusted to full frame coordinates
        """
        # Extract ROI from frame
        roi_frame = extract_roi_from_frame(frame, roi_coords)
        
        # Run detection on cropped frame
        roi_detections = detection_function(roi_frame)
        
        # Adjust coordinates back to full frame
        full_frame_detections = adjust_detections_to_full_frame(roi_detections, roi_coords)
        
        return full_frame_detections
    
    return process_frame_with_roi


def get_roi_performance_stats(roi_coords: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> dict:
    """
    Calculate performance statistics for ROI processing.
    
    Args:
        roi_coords (Tuple[int, int, int, int]): ROI coordinates
        frame_shape (Tuple[int, int]): Frame shape (height, width)
    
    Returns:
        dict: Performance statistics
    """
    x_min, y_min, x_max, y_max = roi_coords
    height, width = frame_shape
    
    roi_width = x_max - x_min
    roi_height = y_max - y_min
    
    full_frame_pixels = height * width
    roi_pixels = roi_height * roi_width
    
    pixel_reduction = (full_frame_pixels - roi_pixels) / full_frame_pixels
    
    return {
        'full_frame_size': (width, height),
        'roi_size': (roi_width, roi_height),
        'pixel_reduction_ratio': pixel_reduction,
        'theoretical_speedup': 1 / (1 - pixel_reduction) if pixel_reduction > 0 else 1.0,
        'roi_area_percentage': (roi_pixels / full_frame_pixels) * 100
    }