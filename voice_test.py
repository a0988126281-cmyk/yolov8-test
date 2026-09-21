import sounddevice as sd
import numpy as np
import speech_recognition as sr
import pyttsx3

# 1. 初始化語音合成引擎 (TTS)
engine = pyttsx3.init()
engine.setProperty('rate', 170)

def speak(text):
    """讓機器人開口說話"""
    print(f"🤖 機器人回應: {text}")
    engine.say(text)
    engine.runAndWait()

def listen(duration=4, sample_rate=16000):
    """使用 sounddevice 錄製麥克風聲音並進行語音辨識 (STT)"""
    recognizer = sr.Recognizer()
    print(f"\n🎤 機器人正在聽... 請對著麥克風說話（錄音時間 {duration} 秒）：")
    
    try:
        # 使用 sounddevice 進行錄音
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # 等待錄音結束
        
        print("⏳ 正在辨識您的語音...")
        
        # 將聲音資料轉換為 SpeechRecognition 可讀取的 AudioData 格式
        audio_bytes = audio_data.tobytes()
        sr_audio = sr.AudioData(audio_bytes, sample_rate, 2)  # 16-bit PCM (2 bytes)
        
        # 使用 Google 辨識引擎轉換成中文
        text = recognizer.recognize_google(sr_audio, language="zh-TW")
        print(f"👤 你說了: {text}")
        return text
        
    except sr.UnknownValueError:
        print("❌ 無法辨識您的聲音，請再試一次。")
        return None
    except sr.RequestError:
        print("❌ 網路連線異常，無法使用語音辨識。")
        return None
    except Exception as e:
        print(f"❌ 麥克風錄音失敗: {e}")
        return None

# --- 主程式 ---
if __name__ == "__main__":
    speak("你好！我是居家照護機器人，請跟我說話。")
    
    while True:
        user_input = listen(duration=4)
        
        if user_input:
            if "你好" in user_input or "嗨" in user_input:
                speak("你好呀！請問有什麼我可以幫忙的？")
            elif "尋找" in user_input or "找" in user_input:
                speak("好的，正在啟動鏡頭為您尋找周圍掉落的物品。")
            elif "再見" in user_input or "結束" in user_input:
                speak("好的，系統即將關閉，再見！")
                break
            else:
                speak(f"我聽到你說了：{user_input}。")