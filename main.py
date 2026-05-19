import os
import sys
from dotenv import load_dotenv

# 讀取 .env 中的 HF_TOKEN
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN or HF_TOKEN == "your_hugging_face_token_here":
    print("錯誤：請先在 .env 檔案中填寫正確的 Hugging Face Token (HF_TOKEN)")
    print("可從這裡獲取：https://huggingface.co/settings/tokens")
    sys.exit(1)

# 延遲載入模型模組，避免一開始就報錯
from audio_recorder import AudioRecorder
from transcriber import Transcriber
from diarizer import Diarizer
from utils import merge_asr_and_diarization

print("系統初始化中...")
recorder = AudioRecorder(hf_token=HF_TOKEN)
transcriber = Transcriber()
diarizer = Diarizer(hf_token=HF_TOKEN)
print("系統初始化完成！")

# 語者對應表 (允許使用者將自動標記的 SPEAKER_00 改為真實姓名)
SPEAKER_MAP = {
    "SPEAKER_00": "Alice",
    "SPEAKER_01": "Bob",
    "SPEAKER_02": "Charlie"
}

def process_audio_chunk(audio_array):
    print("\n--- 偵測到語音段落，開始處理 ---")
    try:
        # 1. 轉錄 (ASR)
        asr_segments = transcriber.transcribe(audio_array)
        
        # 2. 語者分離 (Diarization)
        diarization_results = diarizer.diarize(audio_array)
        
        # 3. 合併對齊
        transcript_lines = merge_asr_and_diarization(asr_segments, diarization_results, SPEAKER_MAP)
        
        # 輸出結果
        for line in transcript_lines:
            print(line)
            
    except Exception as e:
        print(f"處理語音時發生錯誤: {e}")
    print("--------------------------------\n")

if __name__ == "__main__":
    try:
        print("\n準備就緒。對麥克風說話即可觸發辨識。")
        print("提示：程式會偵測靜音 (預設 1.5 秒) 自動截斷並處理。按 Ctrl+C 結束。")
        recorder.process_stream(callback=process_audio_chunk)
    except KeyboardInterrupt:
        print("\n程式結束。")
