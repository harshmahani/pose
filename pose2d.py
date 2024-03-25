import cv2
import mediapipe as mp

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Read video
input_video_path = r"C:\Users\harsh\Downloads\input2d.mp4"
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'XVID'
output_video_path = r"C:\Users\harsh\Downloads\output2d.mp4"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = pose.process(rgb_frame)

    # Extract pose landmarks
    if results.pose_landmarks:
        # Access pose landmarks (results.pose_landmarks)
        landmarks = results.pose_landmarks.landmark

        # Render the output
        for landmark in landmarks:
            h, w, c = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # Example: Draw a circle at each landmark

        # Draw lines connecting pose landmarks
        for connection in mp_pose.POSE_CONNECTIONS:
            start_point = connection[0]
            end_point = connection[1]

            start_landmark = landmarks[start_point]
            end_landmark = landmarks[end_point]

            start_x, start_y = int(start_landmark.x * width), int(start_landmark.y * height)
            end_x, end_y = int(end_landmark.x * width), int(end_landmark.y * height)

            cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # Save frame to the output video
    out.write(frame)

    cv2.imshow('2D Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
