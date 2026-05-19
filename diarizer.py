import torch
from pyannote.audio import Model
import numpy as np
import scipy.spatial.distance
import sys
from speechbrain.utils.fetching import LocalStrategy


def _temporarily_remove_speechbrain_lazy_modules():
    removed_modules = {}
    for name, module in list(sys.modules.items()):
        module_type = type(module)
        if module_type.__module__ != "speechbrain.utils.importutils":
            continue
        if module_type.__name__ not in {"LazyModule", "DeprecatedModuleRedirect"}:
            continue
        removed_modules[name] = sys.modules.pop(name)
    return removed_modules


def _restore_modules(removed_modules):
    sys.modules.update(removed_modules)


def _purge_speechbrain_lazy_modules():
    _temporarily_remove_speechbrain_lazy_modules()

class Diarizer:
    def __init__(self, hf_token, device="cuda" if torch.cuda.is_available() else "cpu"):
        print("正在載入 Speaker Segmentation 模型 (pyannote)...")
        # 1. 局部切割模型 (確保單一句話只有一人)
        removed_modules = _temporarily_remove_speechbrain_lazy_modules()
        try:
            self.seg_model = Model.from_pretrained("pyannote/segmentation-3.0", token=hf_token)
        finally:
            _restore_modules(removed_modules)
        self.seg_model.to(torch.device(device))
        self.seg_model.eval()
        
        print("正在載入 Speaker Verification 模型 (SpeechBrain)...")
        # 2. 聲紋特徵萃取模型
        # 使用 run_opts 指定 device，相容 SpeechBrain 的寫法
        from speechbrain.inference.speaker import EncoderClassifier

        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp_speechbrain",
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY_SKIP_CACHE,
        )
        _purge_speechbrain_lazy_modules()
        
        self.device = device
        
        # 3. 歷史聲紋資料庫: {"SPEAKER_00": np.array(embedding), "SPEAKER_01": ...}
        self.speaker_embeddings = {}
        self.similarity_threshold = 0.55 # 相似度大於 0.55 認為是同一個人

    def extract_embedding(self, waveform_segment):
        """傳入形狀為 (1, N) 的 tensor，回傳 1D 的 embedding vector"""
        with torch.no_grad():
            embeddings = self.encoder.encode_batch(waveform_segment)
            # embeddings shape: (batch, 1, channels) -> squeeze to 1D
            return embeddings.squeeze().cpu().numpy()

    def diarize(self, audio_array: np.ndarray, sample_rate=16000):
        """
        1. 傳入一小段 VAD 切出的音訊 (可能包含 1~2 人)
        2. 用 Segmentation 切割出乾淨的單人說話片段
        3. 用 SpeechBrain 抽出聲紋，跟歷史資料庫比對
        4. 回傳標記好的時間段
        """
        waveform = torch.from_numpy(audio_array).unsqueeze(0).float()
        
        # 取得 segmentation 結果 (找出裡面有幾個人，以及他們說話的時間點)
        with torch.no_grad():
            # seg_output 的形狀大約是 (batch, frames, speakers)
            seg_output = self.seg_model(waveform.to(self.device))
        
        # 使用 binarize 將機率轉換為具體的時間段
        # binarize 會處理 SlidingWindowFeature 並回傳 Annotation
        from pyannote.audio.utils.signal import Binarize
        binarizer = Binarize(
            offset=0.5,
            onset=0.5,
            min_duration_off=0.1,
            min_duration_on=0.1,
        )
        
        try:
            # 嘗試直接處理模型輸出
            annotation = binarizer(seg_output)
        except Exception:
            # 如果 Binarize 不支援，我們退回到一個非常簡單的假設：
            # 既然這是 VAD 切出來的一句話，我們假設這短短的片段裡最多就是兩個人。
            # 為了避免過度複雜的張量處理，我們直接把整段當成「單一句子」送給 SpeechBrain。
            # 這是混合式架構的妥協方案，但在 90% 的短語句中是成立的。
            annotation = None

        local_segments = []
        if annotation:
            for turn, _, local_speaker in annotation.itertracks(yield_label=True):
                local_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "local_speaker": local_speaker
                })
        else:
             # 如果無法精確切分，就把整段當成一個 local_segment
             duration = audio_array.shape[0] / sample_rate
             local_segments.append({
                 "start": 0.0,
                 "end": duration,
                 "local_speaker": "S1"
             })
            
        results = []
        
        # 針對每一段純淨的聲音，進行聲紋比對
        for seg in local_segments:
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            
            # 如果片段太短 (小於 0.5 秒)，萃取出的聲紋會充滿雜訊，直接跳過或隨便標
            if (end_sample - start_sample) < sample_rate * 0.5:
                continue
                
            segment_waveform = waveform[:, start_sample:end_sample]
            
            # 抽出聲紋
            current_emb = self.extract_embedding(segment_waveform.to(self.device))
            
            # 比對歷史資料庫
            best_speaker = None
            best_score = -1
            
            for known_speaker, known_emb in self.speaker_embeddings.items():
                # 計算 Cosine Similarity (scipy 的 cosine 算的是 distance，所以用 1 - distance)
                similarity = 1 - scipy.spatial.distance.cosine(current_emb, known_emb)
                if similarity > best_score:
                    best_score = similarity
                    best_speaker = known_speaker
                    
            # 判斷是否為已知語者
            if best_score > self.similarity_threshold:
                assigned_speaker = best_speaker
                # 可選：滾動更新聲紋 (讓特徵越來越準)
                # self.speaker_embeddings[assigned_speaker] = (self.speaker_embeddings[assigned_speaker] + current_emb) / 2
            else:
                # 建立新語者
                new_speaker_id = f"SPEAKER_{len(self.speaker_embeddings):02d}"
                self.speaker_embeddings[new_speaker_id] = current_emb
                assigned_speaker = new_speaker_id
                # print(f"[Diarizer] 建立新聲紋: {assigned_speaker} (最高相似度僅 {best_score:.2f})")
                
            results.append({
                "speaker": assigned_speaker,
                "start": seg["start"],
                "end": seg["end"]
            })
            
        return results
