def merge_asr_and_diarization(asr_segments, diarization_results, speaker_map=None, punc_restorer=None):
    """
    將 ASR 的時間段與 Diarization 的時間段進行比對，
    找出每一段 ASR 文字對應的說話者。
    """
    if speaker_map is None:
        speaker_map = {}
        
    final_transcript = []
    
    for segment in asr_segments:
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        text = segment.get("text", "").strip()
        
        if not text:
            continue
            
        if punc_restorer:
            text = punc_restorer.restore(text)
            
        # 尋找重疊時間最長的 speaker
        max_overlap = 0
        assigned_speaker = "UNKNOWN"
        
        for d in diarization_results:
            overlap_start = max(start_time, d["start"])
            overlap_end = min(end_time, d["end"])
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                assigned_speaker = d["speaker"]
                
        # 替換為別名 (如果存在)，否則保留原標籤 (如 SPEAKER_00)
        display_speaker = speaker_map.get(assigned_speaker, assigned_speaker)
        
        final_transcript.append(f"[{display_speaker}] {text}")
        
    return final_transcript


class PunctuationRestorer:
    def __init__(self, model_name='p208p2002/zh-wiki-punctuation-restore', device='cpu'):
        print(f"正在載入標點修復模型: {model_name}...")
        try:
            from transformers import pipeline
            device_id = 0 if device == 'cuda' else -1
            self.punc_pipeline = pipeline('token-classification', model=model_name, device=device_id)
            print("標點修復模型載入完成。")
        except Exception as e:
            print(f"標點修復模型載入失敗: {e}")
            self.punc_pipeline = None

    def restore(self, text):
        if not self.punc_pipeline or not text.strip():
            return text
            
        try:
            results = self.punc_pipeline(text)
            
            # 如果沒有預測到任何標點，直接回傳原字串
            if not results:
                return text
                
            # 將結果根據 start index 排序
            results = sorted(results, key=lambda x: x['start'])
            
            restored_text = ""
            last_idx = 0
            
            for res in results:
                entity = res['entity']
                end_idx = res['end']
                
                # 將上一個標點到這個標點之間的「原始文字」原封不動地加進來
                restored_text += text[last_idx:end_idx]
                
                # 加上預測的標點
                if entity not in ['O', '0'] and '-' in entity:
                    punc = entity.split('-')[-1]
                    restored_text += punc
                    
                last_idx = end_idx
                
            # 把最後一個標點之後的剩餘文字加上去
            restored_text += text[last_idx:]
            
            return restored_text
        except Exception as e:
            print(f"標點修復失敗: {e}")
            return text
