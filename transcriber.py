import whisperx
import torch
import numpy as np
from whisperx.diarize import Segment
from whisperx.vads.vad import Vad


class FullAudioVad(Vad):
    def __init__(self, vad_onset=0.5):
        super().__init__(vad_onset)

    @staticmethod
    def preprocess_audio(audio):
        return audio

    def __call__(self, audio, **kwargs):
        sample_rate = audio["sample_rate"]
        waveform = audio["waveform"]
        duration = waveform.shape[-1] / sample_rate
        return [Segment(0.0, duration, "UNKNOWN")]

class Transcriber:
    def __init__(self, model_name="breeze-asr-25-ct2", device="cuda"):
        print(f"正在載入 ASR 模型: {model_name}...")
        self.device = device if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if self.device == "cuda" else "int8"
        
        # 載入 Breeze ASR 模型
        self.model = whisperx.load_model(
            model_name,
            self.device,
            compute_type=compute_type,
            language="zh",
            vad_model=FullAudioVad(),
        )
        
        # 載入對齊模型 (中文)
        print("正在載入字詞對齊模型...")
        self.align_model, self.align_metadata = whisperx.load_align_model(language_code="zh", device=self.device)
        print("ASR 與對齊模型載入完成。")

    def transcribe(self, audio_array: np.ndarray, sample_rate=16000):
        """
        轉換音訊為文字，並進行時間軸對齊
        :param audio_array: 1D numpy array
        :return: 包含對齊後時間戳記的段落 (segments)
        """
        audio = audio_array.astype(np.float32)
        
        # 1. 轉錄
        result = self.model.transcribe(audio, batch_size=1)
        
        # 2. 對齊 (提供更精確的詞級時間戳記)
        result = whisperx.align(result["segments"], self.align_model, self.align_metadata, audio, self.device, return_char_alignments=False)
        
        return result["segments"]
