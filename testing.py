import cv2
import os

# Path to folder containing JPG frames
folder_path = "frames"  # change this to your folder

# Frame rate (frames per second)
fps = 30
delay = int(1000 / fps)  # delay in milliseconds

# Get sorted list of image files
image_files = sorted([
    f for f in os.listdir(folder_path)
    if f.lower().endswith(".jpg")
])

if not image_files:
    print("No JPG files found in folder.")
    exit()

# Read first frame to get window size
first_frame = cv2.imread(os.path.join(folder_path, image_files[0]))
height, width, _ = first_frame.shape

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", width, height)

# Play frames
for img_name in image_files:
    img_path = os.path.join(folder_path, img_name)
    frame = cv2.imread(img_path)

    if frame is None:
        continue

    cv2.imshow("Video", frame)

    # Wait and allow exit on 'q'
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()