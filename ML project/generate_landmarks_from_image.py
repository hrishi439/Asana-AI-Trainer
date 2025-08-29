import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

input_folder = "static/poses"
output_folder = "pose_landmarks"
os.makedirs(output_folder, exist_ok=True)

# Sort input JPGs by the number in their filename, e.g. pose1.jpg, pose2.jpg, … pose12.jpg
jpg_files = [
    f for f in os.listdir(input_folder)
    if f.lower().endswith(".jpg")
]
jpg_files = sorted(
    jpg_files,
    key=lambda x: int(x.replace("pose", "").replace(".jpg", ""))
)

for filename in jpg_files:
    # now filename goes pose1.jpg, pose2.jpg, …, pose12.jpg

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            save_path = os.path.join(output_folder, filename.replace(".jpg", ".npy"))
            np.save(save_path, landmarks)
            print(f"✅ Saved {save_path}")
        else:
            print(f"❌ No pose detected in {filename}")