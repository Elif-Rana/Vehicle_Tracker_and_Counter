## **Project Overview**

#### **Description**

This is a Multi-Line Vehicle Counter that uses computer vision and object tracking to count vehicles crossing multiple detection lines in video footage. The system automatically analyzes vehicle movement patterns and places optimal counting lines to track vehicle crossings. Scripts are suitable for running on both CPU and GPU.

#### **Project Structure**

```
utils/
├── line_utils.py                   # Utilities for line placement and flow analysis
├── line_crossing_tracker.py        # Enhanced tracker for line crossing detection
├── multiline_vehicle_counter.py    # Main counting logic and video processing
└── roi_utils.py                    # Interactive ROI applier compoent
app.py                              # Command-line interface
README.md                           # Project documentation
requirements.txt                    # Python dependencies
```

#### **Project Workflow**

A general project workflow is as follows:

- Get parameters including input video path from user
- Initialize ByteTrack for vehicle tracking
- Anaylze first frames (200 as the number of frames were given hard-coded) to detect vehicle movements and determine dominant traffic flow direction (horizontal or vertical).
- Create evenly spaced counting lines perpendicular to the dominant traffic flow. (For n lines, divide the video into (n+1) equal segments with lines at boundaries.)
- Initialize LineCrossingTracker to monitor vehicle positions relative to each counting line.
- Process each video frame through detection, tracking, crossing detection, and annotation. (YOLOv11n was written as hard-coded to detect vehicle classes, the reason behind chossing YOLOv11n was that it promised to provide better results in the detection area in a shorter time by having fewer parameters (fewer parameter is also an effect to chose it instead of other YOLOv11 models) than previous model versions. And the reason why YOLOv12 was not selected is that there was no any speed information on CPU that is provided on the original website.)
    - Assign consistent tracter IDs using ByteTrack
    - Add bounding boxes, labels, and counting lines
    - Calculate vehicle position relative to counting lines
    - Identify when vehicles cross lines and determine direction
    - Record crossings and update counters dynamically
- Apply processing callback to frames and generate annotated output video
- Calculate final statistics including total unique vehicles, per-line crossing counts and average frame processing time

**Data Flow Summary:**
Input Video -> Flow Analysis -> Line Creation -> Frame Processing -> Output Generation


## **SOME ENHANCED FEATURES**


#### **Traffic Flow Tracker**
A traffic flow tracker was added that checks the initial portion of the video to detect the trajectory of the vehicle movement to decide how to position counting lines (horizontal or vertical).

You can check the videos to see the placement of lines:
- *results\result_test4_skip0_yolov11n_3lines_cpu.mp4* --> vertical lines
- *results\result_highway_skip0_yolov11n_cpu_3lines.mp4* --> horizontal lines


#### **Multi-Line Detection System**

Multi line option was added which will separate the video equal parts aiming not to miss any vehicle. This feature is particularly useful for the videos where vehicles start in a position that will not pass the line ever. While benefiting for decreasing the chance to miss any vehicle, it also does not cause a remarkable time increase.

You can make observation by comparing:
- *results\result_vehicles_skip0_yolov11n_cpu_1line.mp4* 
- *results\result_vehicles_skip0_yolov11n_cpu_3lines.mp4*.
Statistics for these videos take part in:
- *outputs\output_vehicles_skip0_yolov11n_cpu_1line.txt* 
- *outputs\output_vehicles_skip0_yolov11n_cpu_3lines.txt*


#### **Line Crossing Tracker**

A line crossing tracker was added to avoid duplicated counts. Even though a vehicle passes through more than one line in a video and both lines count it, the line crossing tracker provides to get unique vehicles to be counted only.

This ensures that vehicles are only counted once, regardless of how many detection lines they cross during their journey through the video frame.


#### **Frame Skip Parameter**

A parameter is used to decide the number of frames to skip after processing one. The parameter name is frame_skip and takes an integer value. 0 means processing all the frames. This operation surely speeds up the processing time. However, if the frame count is not used carefully, it is possible to miss some vehicles. The point to be considered here is the speed of the vehicle relative to the video.

To observe the average process time you can check:

- *output_highway_skip0_yolov11n_cpu_3lines.txt* (no frame skipping)
- *output_highway_skip2_yolov11n_cpu_3lines.txt* (skip 2 frames) --> similar result with smaller process time
- *output_highway_skip3_yolov11n_cpu_3lines.txt* (demonstrates the risk when not using carefully)

#### **Interactive ROI**

Since in some videos, area that need to be paid attention is smaller than the whole area that the video covers, ensuring the model to focus only on the important part saves time. For this aim, an interactive ROI performer was added to the code. The parameter name is use_roi, and to apply interactive roi, it is enough to make sure it places while running the script.

For compare the results, you can check:
- *outputs\output_test13_skip0_yolov11n_cpu_3lines.txt*
- *outputs\output_test13_skip0_yolov11n_cpu_3lines_roi.txt*
Video results of related statistics:
- *results\result_test13_skip0_yolov11n_cpu_3lines.mp4*
- *results\result_test13_skip0_yolov11n_cpu_3lines_roi.mp4*

You can also check the accuracy of the results obtained after applying ROI to the video below, taken from a distance and from above (12th floor).
- *results\result_test_roi_skip0_yolov11n_cpu_1line.mp4* --> without ROI
- *results\result_test_roi_skip0_yolov11n_cpu_1line_roi.mp4* --> with ROI


##### **Restriction**
In cases where the vision model (YOLOv11n) does not perform well (insufficient light, shooting from a far distance, etc.) and in videos where there are obstacles in front of the vehicles - which causes the labels of the vehicle before and after the obstacle to be assigned differently - the accuracy rate decreases.

In *results\result_test7_skip0_yolov11n_cpu_3lines_roi.mp4", you can see an example for obstacle issue. Both the plants between the lanes and the visuals of cars passing in front of each other prevented a good performance.

## **HOW TO RUN?**

#### **Prerequisites**
- Python 3.11

#### **Setup**

- Check Python Version

```
python --version
# Should show Python 3.11.x
```

- Create and Activate Virtual Environment

```
# Create virtual environment with Python 3.11
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

- Install Dependencies

```
pip install -r requirements.txt
```

#### **Running the Script**

- Basic Usage

```
# Required parameters only (num_lines defaults to 3, frame_skip defaults to 0)
python app.py --input_video_path <path_to_video> --output_video_path <output_path> --stats_output_path <stats_path>

# With optional num_lines parameter
python app.py --input_video_path <path_to_video> --output_video_path <output_path> --stats_output_path <stats_path> --num_lines <number> --frame_skip <number> --use_roi
```

- Custom Usage

```
# python app.py --input_video_path test_videos\test1.mp4 --output_video_path results/result_test1_multiline.mp4 --stats_output_path outputs/output_test1_multiline.txt --num_lines 3
```

