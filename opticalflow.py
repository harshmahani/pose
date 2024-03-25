import cv2
import numpy as np

def calculate_optical_flow(prev_frame, current_frame):
    # Convert frames to grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame_gray, current_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Compute the magnitude of the optical flow vectors
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # Normalize the magnitude to the range [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8 for grayscale
    optical_flow = magnitude.astype(np.uint8)

    return optical_flow

def process_video(input_video_path, output_video_path):
    video = cv2.VideoCapture(input_video_path)

    # Get the first frame to initialize dimensions
    ret, first_frame = video.read()
    height, width, _ = first_frame.shape

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height),
                          isColor=False)  # Set isColor to False for grayscale

    prev_frame = None

    while True:
        # Read the current frame
        ret, current_frame = video.read()
        if not ret:
            break

        # If it's the first frame, set it as the previous frame
        if prev_frame is None:
            prev_frame = current_frame
            continue

        # Calculate optical flow
        optical_flow_frame = calculate_optical_flow(prev_frame, current_frame)

        # Display or save the frame
        out.write(optical_flow_frame)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Update the previous frame
        prev_frame = current_frame

    video.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
input_video_path = "C:\\Users\\harsh\\Downloads\\input2d.mp4"
output_video_path = "C:\\Users\\harsh\\Downloads\\output4.mp4"

process_video(input_video_path, output_video_path)
