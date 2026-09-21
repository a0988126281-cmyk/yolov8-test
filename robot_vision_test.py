import cv2
from ultralytics import YOLO

# 1. 載入官方預訓練的 YOLOv8 輕量版模型
model = YOLO("yolov8n.pt")

# 2. 開啟筆電的內建攝影機
cap = cv2.VideoCapture(0)

print("正在啟動攝影機... (請在跳出的視窗中按下 'q' 鍵來關閉)")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("無法讀取攝影機畫面，請確認攝影機是否被其他程式佔用。")
        break

    # 3. 進行物件偵測 (這裡將信心度調高到 0.5，分數高於 50% 才顯示，減少誤判)
    results = model(frame, conf=0.5)

    # 4. 處理偵測結果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 取得邊界框的座標
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            # 【修改重點 1】取得模型預測的物品類別編號 (class id)
            class_id = int(box.cls[0])
            
            # 【修改重點 2】透過模型內建的字典 (model.names)，把編號轉換成英文名稱
            class_name = model.names[class_id]

            # 繪製邊界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 【修改重點 3】將標籤改為顯示真實的物品名稱與信心度
            label = f"{class_name}: {confidence:.2f}"
            
            # 在框框上方顯示文字
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 5. 顯示即時畫面
    cv2.imshow("Robot Vision Test (Press 'q' to exit)", frame)

    # 6. 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()