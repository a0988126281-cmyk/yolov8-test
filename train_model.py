from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 載入基礎預訓練模型
    model = YOLO("yolov8n.pt")

    print("🚀 開始訓練新資料集...")

    # 2. 開始訓練
    results = model.train(
        # 📌 1. 請把這裡改為你「新解壓縮的資料夾名稱」
        data="C:/python_test/Indoor-objects.v1i.yolov8/data.yaml", 
        
        epochs=30,          # 訓練輪數（若圖片較多可以設 30~50 輪）
        imgsz=640,
        device="cpu",
        
        # 📌 2. 建議加上這個參數！(給這次訓練取個名字)
        # 這樣訓練出來的模型就會固定存放在 runs/detect/indoor_model/weights/best.pt
        # 不會跟之前的 train、train2 混在一起！
        name="indoor_model"  
    )

    print("✅ 訓練完成！模型已儲存在 runs/detect/indoor_model/weights/best.pt")