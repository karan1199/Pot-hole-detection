import cv2
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="1Ps3N7KAj9M7Kn0X0NUG")

# Load the projects and models
project = rf.workspace().project("ipd-pothole-detection")
model = project.version(7).model

project2 = rf.workspace().project("crack-and-dent-detection")
model2 = project2.version(3).model

# Video input path
input_video_path = "10.mp4"
output_video_path = "10_out.mp4"

# Video capture object
cap = cv2.VideoCapture(input_video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

def inter(predictions,frame_rgb):
    for bounding_box in predictions["predictions"]:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2
        class_name = bounding_box['class']
        confidence = bounding_box['confidence']
        # position coordinates: start = (x0, y0), end = (x1, y1)
        # color = RGB-value for bounding box color, (0,0,0) is "black"
        # thickness = stroke width/thickness of bounding box
        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        # draw/place bounding boxes on image
        cv2.rectangle(frame_rgb, start_point, end_point, color=(0,0,0), thickness=2)

        (text_width, text_height), _ = cv2.getTextSize(
            f"{class_name} | {confidence}",
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2)

        cv2.rectangle(frame_rgb, (int(x0), int(y0)), (int(x0) + text_width, int(y0) - text_height), color=(0,0,0),
            thickness=-1)
        
        text_location = (int(x0), int(y0))
        
        cv2.putText(frame_rgb, f"{class_name} | {confidence}",
                    text_location, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                    color=(255,255,255), thickness=2)
    return frame_rgb

while cap.isOpened():
    ret, frame = cap.read()
    
    # Convert frame to RGB (Roboflow expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference on the current frame
    predictions = model.predict(frame_rgb, confidence=50, overlap=30).json()
    
    # Optionally, you can process the predictions here and draw them on the frame using OpenCV functions
    frame_rgb = inter(predictions,frame_rgb)

    # Convert the frame back to BGR (for OpenCV video writer)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Write the annotated frame to the output video
    out.write(frame_bgr)

    # Display the frame (optional)
    cv2.imshow("Video Player", frame_bgr)

    # Press 'q' to exit the video player
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close the OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()


ROBOFLOW_KEY=1Ps3N7KAj9M7Kn0X0NUG ./infer.sh ipd-pothole-detection/7 9.mp4 9_out.mov

ROBOFLOW_KEY=1Ps3N7KAj9M7Kn0X0NUG ./infer.sh crack-and-dent-detection/3 10out.mp4 1000.mov --confidence 15 --fps_in 45
