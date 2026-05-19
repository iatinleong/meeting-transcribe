import os
import queue
import time
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from df.enhance import enhance as df_enhance, init_df
from pyannote.audio import Pipeline
from pyannote.core import Segment

class AudioRecorder:
    def __init__(self, hf_token, sample_rate=16000, silence_threshold=1.5):
        """
        :param hf_token: Hugging Face Token for pyannote
        :param sample_rate: 音訊取樣率，預設 16000Hz (Whisper 及 pyannote 預設)
        :param silence_threshold: 靜音幾秒後截斷 (秒)
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.buffer = []
        
        from pyannote.audio import Model
        from pyannote.audio.pipelines import VoiceActivityDetection
        
        # 載入 VAD 模型 (改用 segmentation-3.0 以避免新版 pyannote 載入錯誤)
        print("正在載入 VAD 模型...")
        model = Model.from_pretrained(
            "pyannote/segmentation-3.0", 
            token=hf_token
        )
        self.vad_pipeline = VoiceActivityDetection(segmentation=model)
        
        # 設定 VAD 參數
        self.vad_pipeline.instantiate({
            "min_duration_on": 0.1,
            "min_duration_off": 0.1
        })
        
        if torch.cuda.is_available():
            self.vad_pipeline.to(torch.device("cuda"))
        print("VAD 模型載入完成。")

        print("正在載入 DeepFilterNet3...")
        self.df_model, self.df_state, _ = init_df()
        print("DeepFilterNet3 載入完成。")

    def _audio_callback(self, indata, frames, time_info, status):
        """麥克風音訊回調"""
        if status:
            print(f"錄音狀態警告: {status}")
        # 將資料放入 queue
        self.audio_queue.put(indata.copy())

    def process_stream(self, callback):
        """
        開始錄音並處理串流
        :param callback: 當擷取到一段完整的語音時呼叫的函數，傳入 (waveform_numpy_array)
        """
        self.is_recording = True
        print("開始監聽麥克風...")
        
        # 參數設定
        chunk_duration = 0.5 # 每次處理 0.5 秒的資料來判斷 VAD
        chunk_samples = int(self.sample_rate * chunk_duration)
        current_chunk = []
        
        last_speech_time = time.time()
        is_speaking_now = False

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', callback=self._audio_callback):
            while self.is_recording:
                try:
                    data = self.audio_queue.get(timeout=0.1)
                    self.buffer.append(data)
                    current_chunk.append(data)
                    
                    current_length = sum([len(d) for d in current_chunk])
                    if current_length >= chunk_samples:
                        # 將目前的 chunk 轉為 numpy array (1, N) 給 pyannote
                        chunk_np = np.concatenate(current_chunk, axis=0).flatten()
                        waveform = torch.from_numpy(chunk_np).unsqueeze(0)
                        
                        # 執行 VAD
                        vad_results = self.vad_pipeline({"waveform": waveform, "sample_rate": self.sample_rate})
                        
                        has_speech = False
                        for speech in vad_results.get_timeline().support():
                            has_speech = True
                            break
                        
                        if has_speech:
                            last_speech_time = time.time()
                            is_speaking_now = True
                        else:
                            # 如果之前在說話，而且現在靜音超過閾值
                            if is_speaking_now and (time.time() - last_speech_time > self.silence_threshold):
                                # 截斷並送出
                                full_audio = np.concatenate(self.buffer, axis=0).flatten()
                                # DeepFilterNet3 在 48kHz 運作，先升頻再降噪再降回 16kHz
                                audio_t = torch.from_numpy(full_audio).unsqueeze(0)
                                audio_t = torchaudio.functional.highpass_biquad(audio_t, self.sample_rate, cutoff_freq=80.0)
                                audio_48k = torchaudio.functional.resample(audio_t, self.sample_rate, self.df_state.sr())
                                enhanced_48k = df_enhance(self.df_model, self.df_state, audio_48k, atten_lim_db=12)
                                full_audio = torchaudio.functional.resample(enhanced_48k, self.df_state.sr(), self.sample_rate).squeeze(0).numpy()
                                callback(full_audio)
                                
                                # 清空 buffer
                                self.buffer = []
                                is_speaking_now = False
                        
                        current_chunk = []
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    self.stop()
                    break

    def stop(self):
        self.is_recording = False
        print("停止錄音。")
