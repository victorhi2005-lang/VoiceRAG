from sentence_transformers import SentenceTransformer, CrossEncoder
from faster_whisper import WhisperModel, BatchedInferencePipeline
import torch


print("正在載入 Embedding 向量模型 (BAAI/bge-m3)，請稍候...", flush=True)
embeddings_model = SentenceTransformer(
    "BAAI/bge-m3",
    device="cuda",
    model_kwargs={"torch_dtype": torch.float16}
)

print("正在載入 Reranker 精排模型 (BAAI/bge-reranker-v2-m3)，請稍候...", flush=True)
reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    device="cuda",
    model_kwargs={"torch_dtype": torch.float16}
)

print("正在載入本地端 Whisper 模型 (large-v3-turbo)，請稍候...", flush=True)
base_whisper_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
whisper_model = BatchedInferencePipeline(model=base_whisper_model)

print("[OK] 所有 AI 系統與資料庫載入完成！", flush=True)
