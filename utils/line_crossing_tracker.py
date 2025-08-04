from collections import defaultdict
from typing import Tuple, List, Dict, Set
import supervision as sv

class LineCrossingTracker:
    """
    Tracker class that identifies which specific vehicles cross which lines.
    """
    def __init__(self, num_lines: int):
        self.num_lines = num_lines
        self.vehicle_line_positions = {}  # vehicle positions relative to each line -> {tracker_id: {line_id: "side"}}
        self.crossed_vehicles = defaultdict(set)  # which vehicles have crossed which lines -> {tracker_id: {(line_id, direction)}}
        self.line_crossings = {}  # {line_id: {"in": [(tracker_id, class_name)], "out": [...]}}

        for line_id in range(num_lines):
            self.line_crossings[line_id] = {"in": [], "out": []}

    def get_vehicle_side_of_line(self, centroid: Tuple[float, float], line_zone: sv.LineZone, direction: str) -> str:
        """
        Determine which side of the line a vehicle is on.
        
        Args:
            centroid (Tuple[float, float]): Vehicle center coordinates (x, y)
            line_zone (sv.LineZone): Line zone object to check against
            direction (str): Line orientation ("horizontal" or "vertical")
        
        Returns:
            str: Side identifier ("top"/"bottom" for horizontal, "left"/"right" for vertical)
        """
        x, y = centroid

        if direction == "horizontal":
            # For horizontal lines, compare y coordinates
            line_y = line_zone.vector.start.y
            return "top" if y < line_y else "bottom"
        else:
            # For vertical lines, compare x coordinates
            line_x = line_zone.vector.start.x
            return "left" if x < line_x else "right"

    def update_vehicle_positions(self, detections: sv.Detections, counting_lines: List[sv.LineZone], direction: str, class_names: dict) -> List[Tuple[int, int, str, str]]:
        """
        Update vehicle positions and detect line crossings.
        
        Args:
            detections (sv.Detections): Current frame detections with tracking IDs
            counting_lines (List[sv.LineZone]): List of counting line zones
            direction (str): Line orientation ("horizontal" or "vertical")
            class_names (dict): Mapping of class IDs to class names
        
        Returns:
            List[Tuple[int, int, str, str]]: List of (tracker_id, line_id, direction, class_name) 
            for newly crossed vehicles
        """
        newly_crossed = []

        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue

            class_id = detections.class_id[i]
            class_name = class_names[class_id]
            bbox = detections.xyxy[i]
            centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            if tracker_id not in self.vehicle_line_positions:
                self.vehicle_line_positions[tracker_id] = {}

            # position relative to each line
            for line_id, line_zone in enumerate(counting_lines):
                current_side = self.get_vehicle_side_of_line(centroid, line_zone, direction)

                if line_id in self.vehicle_line_positions[tracker_id]:
                    previous_side = self.vehicle_line_positions[tracker_id][line_id]    # previous position for this vehicle on this line

                    if previous_side != current_side:   # Vehicle crossed the line
                        if direction == "horizontal":
                            crossing_direction = "in" if (previous_side == "bottom" and current_side == "top") else "out"
                        else:
                            crossing_direction = "in" if (previous_side == "left" and current_side == "right") else "out"

                        crossing_key = (line_id, crossing_direction)
                        # Save the crossing if not recorded -> prevent duplication
                        if crossing_key not in self.crossed_vehicles[tracker_id]:
                            self.crossed_vehicles[tracker_id].add(crossing_key)
                            self.line_crossings[line_id][crossing_direction].append((tracker_id, class_name))
                            newly_crossed.append((tracker_id, line_id, crossing_direction, class_name))

                self.vehicle_line_positions[tracker_id][line_id] = current_side

        return newly_crossed

    def get_unique_vehicle_count(self) -> int:
        """
        Get count of unique vehicles that have crossed any line.
        
        Returns:
            int: Number of unique vehicles that have crossed at least one line
        """
        return len(self.crossed_vehicles)

    def get_line_stats(self, line_id: int) -> Dict:
        """
        Get statistics for a specific line.
        
        Args:
            line_id (int): ID of the line to get stats for
        
        Returns:
            Dict: Statistics containing in_count, out_count, in_vehicles, out_vehicles
        """
        if line_id in self.line_crossings:
            return {
                "in_count": len(self.line_crossings[line_id]["in"]),
                "out_count": len(self.line_crossings[line_id]["out"]),
                "in_vehicles": self.line_crossings[line_id]["in"],
                "out_vehicles": self.line_crossings[line_id]["out"]
            }
        return {"in_count": 0, "out_count": 0, "in_vehicles": [], "out_vehicles": []}