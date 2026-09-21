import cv2
import time
import threading
import queue
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import pyttsx3
from ultralytics import YOLO

# 語音指令佇列
command_queue = queue.Queue()

# 全域變數：供 AI 執行緒與畫面顯示執行緒共用
latest_detections = []  # 儲存最新的劃框資訊 [(x1, y1, x2, y2, label, color), ...]
latest_detected_names = set()
lock = threading.Lock() # 線程安全鎖
is_running = True

def speak(text):
    """核心語音輸出：重新初始化 pyttsx3 避免 Windows 鎖死"""
    print(f"🤖 機器人: {text}")
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"❌ 語音播放錯誤: {e}")

def speak_async(text):
    """背景語音播放"""
    threading.Thread(target=speak, args=(text,), daemon=True).start()

def voice_listener_loop():
    """背景語音監聽執行緒"""
    recognizer = sr.Recognizer()
    sample_rate = 16000
    duration = 3

    print("🎤 背景語音監聽已啟動...")

    while is_running:
        try:
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()
            
            audio_bytes = audio_data.tobytes()
            sr_audio = sr.AudioData(audio_bytes, sample_rate, 2)
            
            text = recognizer.recognize_google(sr_audio, language="zh-TW")
            if text:
                print(f"\n👤 聽到語音指令: {text}")
                command_queue.put(text)
        except Exception:
            pass

# ====================================================
# 1. 初始化四大視覺模型引擎
# ====================================================
print("🚀 正在初始化多重视覺模型群...")

model_general = YOLO("yolov8n.pt")
model_world = YOLO("yolov8s-world.pt")
world_classes = ["pillow", "tissue box", "glasses", "towel", "slippers", "trash can"]
model_world.set_classes(world_classes)

model_nutella = YOLO("runs/detect/train-2/weights/best.pt")
model_indoor = YOLO("runs/detect/indoor_model/weights/best.pt")

print("✅ 所有 4 個視覺模型均已載入完畢！")

# ====================================================
# 2. AI 背景推論執行緒 (專門負責耗時的 AI 計算)
# ====================================================
current_frame_to_process = None

def ai_inference_loop():
    """在背景獨立運行 4 大 AI 模型，不卡死攝影機畫面渲染"""
    global current_frame_to_process, latest_detections, latest_detected_names, is_running
    
    while is_running:
        if current_frame_to_process is None:
            time.sleep(0.01)
            continue

        # 取得最新一幀影像副本
        frame = current_frame_to_process.copy()
        
        # 取得原圖尺寸
        h, w, _ = frame.shape
        
        # ⚡ 核心加速技巧：將輸入 AI 的影像縮小至 320x320 (運算速度快 4 倍)
        small_frame = cv2.resize(frame, (320, 320))
        scale_x = w / 320.0
        scale_y = h / 320.0

        new_detections = []
        new_names = set()

        # 1. 綠色框：通用 COCO (imgsz=320 縮小計算量)
        res_gen = model_general(small_frame, conf=0.4, imgsz=320, verbose=False)
        for r in res_gen:
            for box in r.boxes:
                name = model_general.names[int(box.cls[0])]
                new_names.add(name)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 放大回原圖座標
                new_detections.append((int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y), f"[General] {name}", (0, 255, 0)))

        # 2. 藍/橘色框：YOLO-World
        res_world = model_world(small_frame, conf=0.35, imgsz=320, verbose=False)
        for r in res_world:
            for box in r.boxes:
                name = model_world.names[int(box.cls[0])]
                new_names.add(name)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                new_detections.append((int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y), f"[World] {name}", (255, 165, 0)))

        # 3. 紫色框：Nutella
        res_nutella = model_nutella(small_frame, conf=0.4, imgsz=320, verbose=False)
        for r in res_nutella:
            for box in r.boxes:
                name = model_nutella.names[int(box.cls[0])]
                new_names.add(name)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                new_detections.append((int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y), f"[Nutella] {name}", (255, 0, 255)))

        # 4. 黃色框：Indoor 房間物體
        res_indoor = model_indoor(small_frame, conf=0.4, imgsz=320, verbose=False)
        for r in res_indoor:
            for box in r.boxes:
                name = model_indoor.names[int(box.cls[0])]
                new_names.add(name)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                new_detections.append((int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y), f"[Indoor] {name}", (0, 255, 255)))

        # 安全地更新全域辨識結果
        with lock:
            latest_detections = new_detections
            latest_detected_names = new_names

# ====================================================
# 3. 主程式：高流暢度 (30 FPS) 畫面顯示主迴圈
# ====================================================
if __name__ == "__main__":
    # 啟動背景語音執行緒
    threading.Thread(target=voice_listener_loop, daemon=True).start()
    
    # 啟動背景 AI 運算執行緒
    threading.Thread(target=ai_inference_loop, daemon=True).start()

    speak_async("極速流暢視覺系統已啟動。")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機！")
        exit()

    window_name = "🤖 照護機器人即時視覺監控 (按下 Q 鍵退出)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 將最新畫面傳給背景 AI 執行緒去計算
        current_frame_to_process = frame

        display_frame = frame.copy()

        # 安全地讀取背景 AI 算好的框框並繪製
        with lock:
            boxes_to_draw = list(latest_detections)
            current_names = set(latest_detected_names)

        for x1, y1, x2, y2, label, color in boxes_to_draw:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 畫面上方狀態欄
        status_text = f"Detected: {', '.join(current_names) if current_names else 'Scanning...'}"
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 以最快速度顯示畫面 (實現高 FPS)
        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            speak("系統即將關閉。")
            break

        # 處理背景語音指令
        if not command_queue.empty():
            user_input = command_queue.get()

            if "你好" in user_input or "嗨" in user_input:
                speak_async("哈囉！我正在即時監控周圍環境，隨時準備為您服務。")

            elif "看到" in user_input or "找" in user_input or "物品" in user_input or "什麼" in user_input:
                if current_names:
                    items_str = "、".join(current_names)
                    speak_async(f"我目前幫您看到了：{items_str}")
                else:
                    speak_async("目前鏡頭前沒有看到明確的目標物品。")

            elif "再見" in user_input or "結束" in user_input or "關閉" in user_input:
                is_running = False
                speak("系統即將關閉，祝您有美好的一天。")
                break

    cap.release()
    cv2.destroyAllWindows()