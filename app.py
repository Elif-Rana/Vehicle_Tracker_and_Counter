import torch
from ultralytics import YOLO
import argparse

from utils.multiline_vehicle_counter import count_vehicles_multiline
from utils.roi_utils import select_roi_from_video

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("yolo11n.pt").to(device)

'''
classes belong to "vehicle" category (model.model.names):
1: 'bicycle',
2: 'car',
3: 'motorcycle',
4: 'airplane',
5: 'bus',
6: 'train',
7: 'truck',
8: 'boat'
'''


vehicle_ids = [1, 2, 3, 4, 5, 6, 7, 8]
# vehicle_classes = ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']


def main():
    parser = argparse.ArgumentParser(description='Count vehicles in video')
    
    parser.add_argument('--input_video_path', required=True, help='Path to input video file')
    parser.add_argument('--output_video_path', required=True, help='Path to output video file')
    parser.add_argument('--stats_output_path', required=True, help='Path to statistics output file')
    parser.add_argument('--num_lines', type=int, default=3, help='Number of lines (default: 3)')

    parser.add_argument('--use_roi', action='store_true', help='Enable interactive ROI selection')
    parser.add_argument('--frame_skip', type=int, default=0, help='Number of frames to skip between processing')
    
    args = parser.parse_args()
    
    roi_coords = None
    
    # Interactive ROI selection if requested (always uses first frame)
    if args.use_roi:
        print("Starting interactive ROI selection using first frame...")
        roi_coords = select_roi_from_video(args.input_video_path)
        
        if roi_coords:
            print(f"ROI selected: {roi_coords}")
        else:
            print("ROI selection cancelled. Processing entire frame.")

    # Call main vehicle counter function with parsed arguments
    count_vehicles_multiline(
        model=model, vehicle_ids=vehicle_ids,
        input_video_path=args.input_video_path,
        output_video_path=args.output_video_path,
        stats_output_path=args.stats_output_path,
        num_lines=args.num_lines,
        frame_skip=args.frame_skip,
        roi_coords=roi_coords
    )

if __name__ == "__main__":
    main()


'''
python app.py --input_video_path test_videos\test_roi.mp4 --output_video_path results\result_test_roi_skip0_yolov11n_cpu_1line.mp4 --stats_output_path outputs\output_test_roi_skip0_yolov11n_cpu_1line.txt --num_lines 3 --use_roi

python app.py --input_video_path test_videos\highway.ts --output_video_path results\result_highway_skip2_yolov11n_cpu_3lines.mp4 --stats_output_path outputs\output_highway_skip2_yolov11n_cpu_3lines.txt --num_lines 3 --frame_skip 2

'''
