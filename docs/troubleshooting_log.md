# VoiceRAG 開發除錯紀錄 (Troubleshooting Log)

這份文件記錄了在 VoiceRAG 系統升級過程中，遭遇的特殊 Bug 及其解決方案。這些問題主要集中在 AI 模型的依賴套件 (Dependencies) 以及 Windows 環境下的 GPU 資源調用。

---

## 1. FlagEmbedding 與 Transformers 版本不相容 (Tokenizer 錯誤)

### 🚨 錯誤現象
在實作 Phase 5 優化時，系統在執行 `ask_question` 進行問答精排 (Reranking) 時拋出以下錯誤：
```text
TypeError: XLMRobertaTokenizer has no attribute prepare_for_model
```
接著，在嘗試降級 `transformers` 後，又出現以下錯誤：
```text
TypeError: XLMRobertaModel.__init__() got an unexpected keyword argument 'dtype'
```

### 🔍 問題原因
這個問題起因於 `FlagEmbedding` 套件 (提供 BGE-M3 和 Reranker 模型) 與 HuggingFace 的 `transformers` 套件之間的版本夾擊：
1. `transformers` 在最新的 5.x 版本中，移除了 `XLMRobertaTokenizer.prepare_for_model` 這個內部方法，導致 `FlagEmbedding` 舊版本呼叫時報錯。
2. 若將 `transformers` 降級至 4.46 以下，雖然解決了 Tokenizer 問題，但 `FlagEmbedding` 的程式碼在實例化模型時，傳遞了 `dtype` 參數，而舊版 `transformers` 的 `XLMRobertaModel` 並不支援這個參數。

這導致系統陷入了「升級也不對，降級也不對」的死胡同。

### ✅ 解決方案
**徹底棄用 `FlagEmbedding` 的包裝類別，改回使用標準的 `sentence-transformers`。**

由於 BGE-M3 完全相容於標準的 Sentence Transformers 介面，我們將 `main.py` 中的模型載入邏輯從：
```python
# 棄用
from FlagEmbedding import BGEM3FlagModel, FlagReranker
embeddings_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
```
改為：
```python
# 新版
from sentence_transformers import SentenceTransformer, CrossEncoder
embeddings_model = SentenceTransformer('BAAI/bge-m3', device='cuda')
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cuda')
```
這不僅繞過了 `FlagEmbedding` 寫死內部 API 導致的相容性問題，也讓後續的 `.encode()` 和 `.predict()` 呼叫更加簡潔。

---

## 2. Windows 環境下 PyTorch 與 CTranslate2 的無聲崩潰 (Silent Crash)

### 🚨 錯誤現象
在修改完模型載入方式，並透過 `uvicorn main:app --reload` 重新啟動伺服器時，終端機顯示：
```text
INFO:     Started reloader process [12900] using WatchFiles
```
然後游標就卡在那裡，**沒有跳出任何錯誤訊息，也沒有啟動伺服器** (Exit Code 1)。

### 🔍 問題原因
這是 Windows 作業系統下常見的**「動態連結函式庫 (DLL) 衝突 / OpenMP 衝突」**，也被稱為無聲崩潰。

在我們的環境中，有兩個底層極其龐大的 AI 函式庫：
1. **PyTorch** (由 `sentence_transformers` 呼叫，用來跑 BGE-M3)
2. **CTranslate2** (由 `faster_whisper` 呼叫，用來跑 Whisper 語音辨識)

當 Python 讀取 `main.py` 時，如果先 `import faster_whisper`，它會率先在記憶體中載入自己版本的 OpenMP / cuDNN 動態庫；接著當我們 `import sentence_transformers` 並嘗試初始化 PyTorch CUDA 時，PyTorch 發現記憶體中已經有另一個版本的底層函式庫，於是觸發嚴重的衝突。由於衝突發生在 C++ / CUDA 底層，Python 還來不及捕捉並印出 Exception，整個 Process 就直接被作業系統強制終止了。

### ✅ 解決方案
**嚴格控制 `import` 順序，確保 PyTorch 優先取得底層資源。**

我們將 `main.py` 最上方的 `import` 順序進行了調換：

**修改前 (會崩潰)：**
```python
from faster_whisper import WhisperModel
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
```

**修改後 (正常運作)：**
```python
from sentence_transformers import SentenceTransformer, CrossEncoder
from faster_whisper import WhisperModel
import ollama
```

藉由讓 `sentence_transformers` 優先載入，PyTorch 能夠正確初始化其 CUDA 環境，後續再載入 `faster_whisper` 時就能與之和平共存，完美解決了無聲崩潰的問題。同時，我們也補上了 `--force-reinstall` 重新安裝了 `torch-2.11.0+cu128` 以匹配 RTX 5080 的 SM_120 (Compute Capability 12.0) 架構，確保 GPU 加速能滿載運行。
