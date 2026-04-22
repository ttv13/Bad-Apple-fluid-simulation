import cv2
import numpy as np
import os


folder_path = "frames"   
scale = 8               
fps = 30
dt = 1.0 / fps

# parameters
k = 15.0      
damping = 0.85


image_files = sorted([
    f for f in os.listdir(folder_path)
    if f.lower().endswith(".jpg")
])

if not image_files:
    print("No images found.")
    exit()


frame = cv2.imread(os.path.join(folder_path, image_files[0]))
h, w, _ = frame.shape

# Downsampling 
h_s, w_s = h // scale, w // scale

# particle positions and velocities
positions = np.zeros((h_s, w_s, 2), dtype=np.float32)
velocities = np.zeros_like(positions)


for y in range(h_s):
    for x in range(w_s):
        positions[y, x] = np.array([x * scale, y * scale])

cv2.namedWindow("Particle Video", cv2.WINDOW_NORMAL)


for img_name in image_files:
    frame = cv2.imread(os.path.join(folder_path, img_name))
    small = cv2.resize(frame, (w_s, h_s))

    
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) / 255.0

    
    targets = np.zeros_like(positions)
    for y in range(h_s):
        for x in range(w_s):
            base_pos = np.array([x * scale, y * scale])
            offset = np.array([0, -gray[y, x] * scale * 2])
            targets[y, x] = base_pos + offset

    
    force = k * (targets - positions)
    velocities = damping * (velocities + force * dt)
    positions += velocities * dt

    
    canvas = np.zeros_like(frame)

    for y in range(h_s):
        for x in range(w_s):
            px, py = positions[y, x].astype(int)
            if 0 <= px < w and 0 <= py < h:
                canvas[py, px] = small[y, x]

    cv2.imshow("Particle Video", canvas)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()