import re

import numpy as np
import torch
import whisperx
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
    def __init__(self, model_type="breeze", device="cuda"):
        """
        :param model_type: "breeze" (Breeze ASR 2.5) 或 "sensevoice" (SenseVoiceSmall)
        """
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else "cpu"

        if model_type == "sensevoice":
            self._load_sensevoice()
        else:
            self._load_breeze()

    def _load_breeze(self, model_name="breeze-asr-25-ct2"):
        print("正在載入 Breeze ASR 模型...")
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = whisperx.load_model(
            model_name,
            self.device,
            compute_type=compute_type,
            language="zh",
            vad_model=FullAudioVad(),
        )
        print("正在載入字詞對齊模型...")
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="zh", device=self.device
        )
        print("Breeze ASR 載入完成。")

    def _load_sensevoice(self):
        print("正在載入 SenseVoice 模型...")
        from funasr import AutoModel
        self.sv_model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            device=self.device,
        )
        print("SenseVoice 載入完成。")

    def transcribe(self, audio_array: np.ndarray, sample_rate=16000):
        if self.model_type == "sensevoice":
            return self._transcribe_sensevoice(audio_array, sample_rate)
        return self._transcribe_breeze(audio_array)

    def _transcribe_breeze(self, audio_array: np.ndarray):
        audio = audio_array.astype(np.float32)
        result = self.model.transcribe(audio, batch_size=1)
        result = whisperx.align(
            result["segments"], self.align_model, self.align_metadata,
            audio, self.device, return_char_alignments=False,
        )
        return result["segments"]

    def _transcribe_sensevoice(self, audio_array: np.ndarray, sample_rate: int):
        audio = audio_array.astype(np.float32)
        res = self.sv_model.generate(input=audio, cache={}, language="zh", use_itn=True)
        text = re.sub(r"<\|[^|]+\|>", "", res[0]["text"]).strip()
        duration = len(audio) / sample_rate
        return [{"text": text, "start": 0.0, "end": duration}] if text else []
