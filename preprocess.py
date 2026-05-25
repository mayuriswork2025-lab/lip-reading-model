import mediapipe as mp
import cv2
import numpy as np
import os
mpface_mesh=mp.solutions.face_mesh
face_mesh=mpface_mesh.FaceMesh()
lips_landmarks=[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
def extract(video_path, output_path, targetsize=(100,50)):
    cap=cv2.VideoCapture(video_path)
    frames=[]
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            break
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks=results.multi_face_landmarks[0]
            h,w=frame.shape[:2]
            lip_points=[]
            for i in lips_landmarks:
                lm=landmarks.landmark[i]
                lip_points.append((int(lm.x*w),int(lm.y*h)))
                for p in lip_points:
                    x_coord=[p[0]]
                    y_coord=[p[1]]
                    x1,x2=min(x_coord)-10,max(x_coord)+10
                    y1,y2=min(y_coord)-10,max(y_coord)+10
                    mouth=frame[y1:y2,x1:x2]
                    mouth=cv2.resize(mouth,targetsize)
                    mouth=cv2.cvtColor(mouth,cv2.COLOR_BGR2RGB)
                    frames.append(mouth)

    cap.release()
    frames=np.array(frames)
    np.save(output_path, frames)
    print(f"Saved {len(frames)} frames to {output_path}")
    return frames
if __name__=="__main__":
    extract("data/s1/videos/s2_l_bbim3a.mov",
        "data/processed/s2_l_bbim3a.npy"
    )


