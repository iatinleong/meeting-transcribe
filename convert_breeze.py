import os
from huggingface_hub import snapshot_download
import ctranslate2
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

model_id = "MediaTek-Research/breeze-asr-25"
output_dir = "breeze-asr-25-ct2"

print(f"正在下載模型 {model_id}...")
model_path = snapshot_download(repo_id=model_id, token=token)

print(f"正在將模型轉換為 CTranslate2 格式並存儲至 {output_dir}...")
converter = ctranslate2.converters.TransformersConverter(
    model_id, 
    activation_scales=None, 
    load_as_float16=True
)

# 如果目錄已存在則跳過
if not os.path.exists(output_dir):
    converter.convert(output_dir, quantization="float16", force=True)
    print("轉換完成！")
else:
    print(f"目錄 {output_dir} 已存在，跳過轉換。")
