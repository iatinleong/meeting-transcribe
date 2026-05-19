import io
import os
import time

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torchaudio
from df.enhance import enhance as df_enhance, init_df
from dotenv import load_dotenv

from diarizer import Diarizer
from transcriber import Transcriber
from utils import PunctuationRestorer, merge_asr_and_diarization


st.set_page_config(page_title="Meeting Assistant", layout="wide")


@st.cache_resource
def load_base_models():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    model = Model.from_pretrained("pyannote/segmentation-3.0", token=hf_token)
    vad_pipeline = VoiceActivityDetection(segmentation=model)
    vad_pipeline.instantiate({"min_duration_on": 0.1, "min_duration_off": 0.1})

    if torch.cuda.is_available():
        vad_pipeline.to(torch.device("cuda"))

    diarizer = Diarizer(hf_token=hf_token)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    punc_restorer = PunctuationRestorer(device=device_str)
    df_model, df_state, _ = init_df()

    return vad_pipeline, diarizer, punc_restorer, df_model, df_state


@st.cache_resource
def load_transcriber(model_type: str):
    return Transcriber(model_type=model_type)


@st.cache_resource
def load_sepformer():
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.utils.fetching import LocalStrategy
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wham16k-enhancement",
        savedir="pretrained_models/sepformer-wham16k-enhancement",
        run_opts={"device": device_str},
        local_strategy=LocalStrategy.COPY,
    )


st.title("Meeting Assistant - Streaming Demo")
st.markdown(
    "Pipeline: VAD detects speech, diarization splits speakers, "
    "then ASR runs on each final speaker segment."
)

with st.spinner("Loading base models..."):
    vad_pipeline, diarizer, punc_restorer, df_model, df_state = load_base_models()


speaker_map = {
    "SPEAKER_00": "Speaker A",
    "SPEAKER_01": "Speaker B",
    "SPEAKER_02": "Speaker C",
    "SPEAKER_03": "Speaker D",
}

col1, col2 = st.columns([1, 2])

if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""
if "is_simulating" not in st.session_state:
    st.session_state.is_simulating = False

with col1:
    st.subheader("Audio Setup")
    audio_file = st.selectbox(
        "Choose meeting audio",
        [
            "meeting/Songjiang Nanjing Station 28.m4a",
            "meeting/Songjiang Nanjing Station 29.m4a",
        ],
    )

    silence_threshold = st.slider(
        "End-of-speech silence (sec)",
        0.5, 3.0, 1.5, 0.1,
    )
    simulation_speed = st.slider(
        "Simulation speed",
        0.5, 5.0, 1.0, 0.5,
    )

    st.markdown("---")
    st.subheader("ASR Model")
    asr_model = st.selectbox(
        "Speech recognition model",
        ["Breeze ASR 2.5", "SenseVoice"],
        help="Breeze ASR 2.5: Chinese-optimized, word-level timestamps. SenseVoice: faster, supports emotion detection.",
    )

    st.subheader("Enhancement")
    enhancement_model = st.selectbox(
        "Enhancement model",
        ["None", "DeepFilterNet3", "SepFormer"],
        help="None: no processing. DeepFilterNet3: neural noise suppression. SepFormer: speech separation.",
    )
    if enhancement_model == "DeepFilterNet3":
        highpass_cutoff = st.slider(
            "High-pass cutoff (Hz)", 40, 200, 80, 10,
            help="Cut frequencies below this value. Speech starts at ~80 Hz.",
        )
        atten_lim_db = st.slider(
            "DF3 max attenuation (dB)", 6, 30, 12, 1,
            help="Lower = preserve more voice. Higher = more aggressive noise removal.",
        )

    if st.button("Start / Restart", use_container_width=True):
        st.session_state.transcript_text = ""
        st.session_state.is_simulating = True

with col2:
    st.subheader("Transcript")
    transcript_box = st.empty()
    status_box = st.empty()
    transcript_box.text_area(
        "Transcript text",
        st.session_state.transcript_text,
        height=400,
        key="main_box",
    )


def recursive_vad_split(audio_segment: np.ndarray, sr: int, vad_pipeline, current_threshold: float, min_threshold: float = 0.2):
    max_samples = int(sr * 30.0)

    if len(audio_segment) <= max_samples or current_threshold < min_threshold:
        return [audio_segment]

    vad_pipeline.instantiate({
        "min_duration_on": 0.1,
        "min_duration_off": current_threshold
    })

    waveform = torch.from_numpy(audio_segment).unsqueeze(0).float()

    try:
        vad_results = vad_pipeline({"waveform": waveform, "sample_rate": sr})

        segments = []
        for speech in vad_results.get_timeline().support():
            start_sample = int(speech.start * sr)
            end_sample = int(speech.end * sr)

            sub_seg = audio_segment[start_sample:end_sample]
            if len(sub_seg) > max_samples:
                segments.extend(recursive_vad_split(sub_seg, sr, vad_pipeline, current_threshold - 0.3, min_threshold))
            else:
                segments.append(sub_seg)

        vad_pipeline.instantiate({"min_duration_on": 0.1, "min_duration_off": 0.1})

        if not segments:
            return [audio_segment[i:i+max_samples] for i in range(0, len(audio_segment), max_samples)]

        return segments

    except Exception as e:
        print(f"遞迴 VAD 切割失敗: {e}，改用硬切")
        return [audio_segment[i:i+max_samples] for i in range(0, len(audio_segment), max_samples)]


def transcribe_chunk(
    current_audio: np.ndarray,
    sr: int,
    transcriber: Transcriber,
    processed_audio_seconds: float,
    vad_pipeline,
    initial_silence_threshold: float,
):
    asr_segments = []

    if len(current_audio) < int(sr * 0.3):
        return asr_segments

    sub_segments = recursive_vad_split(current_audio, sr, vad_pipeline, initial_silence_threshold - 0.3)

    current_offset_samples = 0
    for sub_audio in sub_segments:
        if len(sub_audio) < int(sr * 0.3):
            current_offset_samples += len(sub_audio)
            continue

        max_samples = int(sr * 30.0)
        for offset in range(0, len(sub_audio), max_samples):
            chunk_to_transcribe = sub_audio[offset:offset + max_samples]

            sub_segment_asr = transcriber.transcribe(chunk_to_transcribe, sample_rate=sr)

            absolute_offset = ((current_offset_samples + offset) / sr) + processed_audio_seconds

            for seg in sub_segment_asr:
                seg["start"] += absolute_offset
                seg["end"] += absolute_offset
                if "words" in seg:
                    for word in seg["words"]:
                        if "start" in word:
                            word["start"] += absolute_offset
                        if "end" in word:
                            word["end"] += absolute_offset
                asr_segments.append(seg)

        current_offset_samples += len(sub_audio)

    return asr_segments


def apply_enhancement(audio_raw: np.ndarray, sr: int, enhancement: str) -> np.ndarray:
    if enhancement == "DeepFilterNet3":
        audio_t = torch.from_numpy(audio_raw).unsqueeze(0)
        audio_t = torchaudio.functional.highpass_biquad(audio_t, sr, cutoff_freq=float(highpass_cutoff))
        audio_48k = torchaudio.functional.resample(audio_t, sr, df_state.sr())
        enhanced_48k = df_enhance(df_model, df_state, audio_48k, atten_lim_db=float(atten_lim_db))
        return torchaudio.functional.resample(enhanced_48k, df_state.sr(), sr).squeeze(0).numpy()

    if enhancement == "SepFormer":
        sepformer = load_sepformer()
        output = audio_raw.copy()

        # Run VAD first to find actual speech segments, then enhance only those
        waveform = torch.from_numpy(audio_raw).unsqueeze(0).float()
        vad_results = vad_pipeline({"waveform": waveform, "sample_rate": sr})
        vad_pipeline.instantiate({"min_duration_on": 0.1, "min_duration_off": 0.1})

        for speech in vad_results.get_timeline().support():
            start = int(speech.start * sr)
            end = int(speech.end * sr)
            segment = audio_raw[start:end]
            if len(segment) < int(sr * 0.1):
                continue
            audio_t = torch.from_numpy(segment).float().unsqueeze(0)
            enhanced = sepformer.separate_batch(audio_t)
            out = enhanced[0, :, 0] if enhanced.dim() == 3 else enhanced[0]
            enhanced_np = out.detach().cpu().numpy()
            # SepFormer may return slightly different length due to encoder padding
            length = min(len(enhanced_np), end - start)
            output[start : start + length] = enhanced_np[:length]

        return output

    return audio_raw


if st.session_state.is_simulating:
    asr_key = "breeze" if asr_model == "Breeze ASR 2.5" else "sensevoice"

    with st.spinner(f"Loading {asr_model}..."):
        transcriber = load_transcriber(asr_key)

    with st.spinner("Loading audio..."):
        audio_data_raw, sr = librosa.load(audio_file, sr=16000)
        audio_data = apply_enhancement(audio_data_raw, sr, enhancement_model)

        peak = np.max(np.abs(audio_data))
        if peak > 0:
            audio_data = (audio_data / peak * 0.95).astype(np.float32)

    with col1:
        st.markdown("---")
        st.write("Original audio")
        st.audio(audio_file)

        st.write("Recognition audio")
        buffer_io = io.BytesIO()
        sf.write(buffer_io, audio_data, sr, format="WAV")
        st.audio(buffer_io, format="audio/wav")

    status_box.success(
        f"Audio loaded. Duration: {len(audio_data) / sr:.2f}s. Starting simulation."
    )

    chunk_duration = 0.5
    chunk_samples = int(sr * chunk_duration)
    buffer = []
    processed_audio_seconds = 0.0
    is_speaking_now = False
    last_speech_time = 0.0

    for i in range(0, len(audio_data), chunk_samples):
        chunk = audio_data[i : i + chunk_samples]
        buffer.append(chunk)
        current_time = (i + len(chunk)) / sr

        status_box.info(
            f"Playback: {current_time:.1f}s | Buffered: {len(buffer) * chunk_duration:.1f}s"
        )
        time.sleep(chunk_duration / simulation_speed)

        waveform = torch.from_numpy(chunk).unsqueeze(0)
        vad_results = vad_pipeline({"waveform": waveform, "sample_rate": sr})
        has_speech = any(True for _ in vad_results.get_timeline().support())

        if has_speech:
            last_speech_time = current_time
            is_speaking_now = True
            status_box.warning(f"Speech detected at {current_time:.1f}s")
            continue

        if not is_speaking_now:
            continue

        if current_time - last_speech_time <= silence_threshold:
            continue

        status_box.error(
            f"Silence exceeded {silence_threshold:.1f}s. Processing chunk."
        )
        current_audio = np.concatenate(buffer, axis=0)

        try:
            asr_segments = transcribe_chunk(
                current_audio=current_audio,
                sr=sr,
                transcriber=transcriber,
                processed_audio_seconds=processed_audio_seconds,
                vad_pipeline=vad_pipeline,
                initial_silence_threshold=silence_threshold,
            )

            lines = []
            for seg in asr_segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                if punc_restorer:
                    text = punc_restorer.restore(text)
                lines.append(f"[Speaker] {text}")

            if lines:
                for line in lines:
                    st.session_state.transcript_text += line + "\n"

                transcript_box.text_area(
                    "Transcript text",
                    st.session_state.transcript_text,
                    height=400,
                    key=f"box_{current_time}",
                )

        except Exception as e:
            print(f"Error: {e}")

        processed_audio_seconds += len(current_audio) / sr
        buffer = []
        is_speaking_now = False

    status_box.success("Simulation completed.")
    st.session_state.is_simulating = False
