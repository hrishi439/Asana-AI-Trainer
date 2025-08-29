import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
import json
from datetime import datetime

PROGRESS_FILE = "progress.json"

# ----------------- Utility Functions -----------------
def load_progress():
    try:
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
            for key in ["dates", "counts", "accuracy", "badges", "history"]:
                if key not in data:
                    data[key] = []
            if "streak" not in data:
                data["streak"] = 0
            if "last_date" not in data:
                data["last_date"] = None
            return data
    except FileNotFoundError:
        return {
            "dates": [],
            "counts": [],
            "accuracy": [],
            "badges": [],
            "history": [],
            "streak": 0,
            "last_date": None
        }

def save_progress(data):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f)

# ----------------- Flask Setup -----------------
app = Flask(__name__)

pose_names = [
    "Pranamasana (Prayer Pose)", "Hasta Uttanasana (Raised Arms)", "Hasta Padasana (Forward Bend)",
    "Ashwa Sanchalanasana (Lunge Right)", "Phalakasana (Plank)", "Ashtanga Namaskara (Eight-Limb Pose)",
    "Bhujangasana (Cobra)", "Adho Mukha Svanasana (Downward Dog)", "Ashwa Sanchalanasana (Lunge Left)",
    "Hasta Padasana (Forward Bend)", "Hasta Uttanasana (Raised Arms)", "Pranamasana (Prayer Pose)"
]

pose_data_folder = "pose_landmarks"
pose_files = sorted(
    [f for f in os.listdir(pose_data_folder) if f.lower().endswith(".npy")],
    key=lambda x: int(x.replace("pose", "").replace(".npy", ""))
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

current_step = 0
highest_score = 0
live_score = 0
pose_locked = False
pose_scores = []  # Stores accuracy of each pose in current session

# ----------------- Utility Logic -----------------
def calculate_similarity(pose1, pose2):
    if pose1.shape != pose2.shape:
        return 0
    dist = np.linalg.norm(pose1 - pose2)
    max_dist = np.linalg.norm(np.ones_like(pose1))
    return max(0, 100 - (dist / max_dist) * 100)

def finalize_pose_score():
    global highest_score, pose_locked, pose_scores
    if highest_score > 0:
        pose_scores.append(int(highest_score))
    pose_locked = True

# ----------------- Camera Stream -----------------
def generate_frames():
    global current_step, highest_score, live_score, pose_locked
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        ref_name = pose_files[current_step]
        reference = np.load(os.path.join(pose_data_folder, ref_name))

        if result.pose_landmarks:
            user_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])
            live_score = calculate_similarity(reference, user_landmarks)
            if not pose_locked:
                highest_score = max(highest_score, live_score)
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        score_display = highest_score if pose_locked else live_score
        color = (0, 128, 255) if pose_locked else (0, 255, 0)
        cv2.putText(frame, f"{pose_names[current_step]} ({current_step + 1}/{len(pose_files)})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f"Score: {int(score_display)}%",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ----------------- Routes -----------------
@app.route("/")
def home():
    progress = load_progress()
    return render_template("home.html", progress=progress)

@app.route('/train')
def train():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/action', methods=['POST'])
def action():
    global current_step, pose_locked, highest_score, live_score, pose_scores
    data = request.json
    action_type = data.get('action')
    step = data.get('step', current_step)
    current_step = step

    if action_type == "stop":
        finalize_pose_score()

    elif action_type == "next":
        if not pose_locked and highest_score > 0:
            finalize_pose_score()
        if current_step < len(pose_files) - 1:
            current_step += 1
        highest_score = 0
        live_score = 0
        pose_locked = False

    elif action_type == "back":
        if current_step > 0:
            current_step -= 1
        highest_score = 0
        live_score = 0
        pose_locked = False

    elif action_type in ["retry", "jump"]:
        highest_score = 0
        live_score = 0
        pose_locked = False

    prog = load_progress()
    total_sessions = sum(prog.get("counts", []))
    best_overall = max(prog.get("accuracy", [0] + [int(highest_score if pose_locked else live_score)]))
    return jsonify({
        "step": current_step,
        "locked": pose_locked,
        "score": int(highest_score if pose_locked else live_score),
        "sessions": total_sessions,
        "best": int(best_overall),
        "streak": prog.get("streak", 0),
        "badges": prog.get("badges", [])
    })

@app.route("/update_progress", methods=["POST"])
def update_progress():
    global pose_scores, highest_score, pose_locked
    try:
        payload = request.get_json() or {}
        count = int(payload.get("count", 1))
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time_now = now.strftime("%H:%M:%S")

        session_accuracy = round(sum(pose_scores)/len(pose_scores),2) if pose_scores else round(float(highest_score),2)

        progress = load_progress()

        # ----------- History (all sessions) -----------
        progress["history"].append({
            "date": date,
            "time": time_now,
            "count": count,
            "accuracy": session_accuracy
        })

        # ----------- Update counts & daily max accuracy -----------
        if date in progress["dates"]:
            idx = progress["dates"].index(date)
            progress["counts"][idx] += count
            prev_acc = progress["accuracy"][idx] if idx < len(progress["accuracy"]) else 0
            progress["accuracy"][idx] = max(prev_acc, session_accuracy)
        else:
            progress["dates"].append(date)
            progress["counts"].append(count)
            progress["accuracy"].append(session_accuracy)

        # ----------- Streak calculation -----------
        unique_days = sorted(set(progress["dates"]))
        streak = 0
        if unique_days:
            streak = 1
            for i in range(len(unique_days)-2,-1,-1):
                d1 = datetime.strptime(unique_days[i],"%Y-%m-%d")
                d2 = datetime.strptime(unique_days[i+1],"%Y-%m-%d")
                if (d2-d1).days == 1:
                    streak += 1
                else:
                    break
        progress["streak"] = streak
        progress["last_date"] = date

        # ----------- Badges -----------
        badge_thresholds = [
            ("aruna", 1),
            ("agni", 7),
            ("gayatri", 14),
            ("arjuna", 21),
            ("nataraja", 30),
            ("aditya", 60),
            ("muralidhara", 90),
            ("suryanatha", 365),
        ]
        badges = set(progress.get("badges", []))
        for bid, days_required in badge_thresholds:
            if streak >= days_required:
                badges.add(bid)
        progress["badges"] = sorted(list(badges))

        save_progress(progress)

        # reset session
        pose_scores = []
        highest_score = 0
        pose_locked = False

        return jsonify({"status":"success","progress":progress})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

@app.route("/progress_data", methods=["GET"])
def progress_data():
    progress = load_progress()
    progress.setdefault("streak",0)
    progress.setdefault("badges",[])
    progress.setdefault("history",[])
    progress.setdefault("dates",[])
    progress.setdefault("counts",[])
    progress.setdefault("accuracy",[])
    return jsonify(progress)

if __name__ == "__main__":
    app.run(debug=True)
