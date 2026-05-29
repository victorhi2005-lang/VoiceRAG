<div align="center">

# 🎙️ VoiceRAG — 智慧語音知識庫系統

**以口述建構知識，以 AI 檢索智慧**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-qwen3.5%3A9b--q4_K_M-8B5CF6?logo=ollama&logoColor=white)](https://ollama.com/)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## 專案總覽

VoiceRAG 將口述錄音轉換為可查詢、可追溯、可長期保存的 AI 知識庫。系統聚焦三件事：保存口述知識、提升檢索效率，並讓 AI 回答能回到原始來源。

```
口述錄音 → Whisper 語音轉文字 → AI 摘要整理 → 切分與向量化 → 向量知識庫 → RAG 智慧問答
```

**核心特色：**
- **本地端優先**：預設使用本機 Ollama 與本機 GPU，音檔、逐字稿與知識庫資料保存在電腦內
- **混合檢索 + 精排**：Dense 向量檢索、BM25、RRF 與 Reranker 逐步精煉結果
- **多筆記本管理**：每個筆記本都是獨立知識空間，來源、對話、向量彼此隔離
- **LLM Provider 可切換**：預設使用 Ollama，也可選用 DeepSeek 進行摘要與問答
- **多輪對話**：自動帶入歷史上下文，支援追問與連續提問
- **來源引用追溯**：AI 回答附帶具體引用來源，可追溯至原始音檔

> 隱私提醒：若選擇 DeepSeek provider，摘要、分析或問答所需的文字內容會送往 DeepSeek API；若需要全本地處理，請維持使用 Ollama provider。

---

## 系統架構

### 技術棧總覽

| 層級 | 技術 | 說明 |
|------|------|------|
| **前端** | HTML5 + JavaScript + TailwindCSS | 單頁式深色介面 |
| **後端** | FastAPI (Python) | API 路由與服務入口 |
| **語音辨識** | faster-whisper (`large-v3-turbo`) | CUDA 加速，VAD 過濾靜音 |
| **Embedding** | `BAAI/bge-m3` (FP16) | 多語言向量化，1024 維 |
| **Reranker** | `BAAI/bge-reranker-v2-m3` (FP16) | CrossEncoder 精排 |
| **LLM** | Ollama `qwen3.5:9b-q4_K_M` / DeepSeek `deepseek-v4-flash` | 摘要、分析、檔名建議與回答生成 |
| **向量資料庫** | ChromaDB | 每筆記本獨立 Collection |
| **關聯資料庫** | SQLite | 筆記本、來源、對話與推薦問題 |
| **關鍵字檢索** | rank_bm25 + jieba | 中文分詞與 BM25 索引 |

### 後端模組分層

目前後端已從單一檔案拆成多個服務模組，讓 API 路由、模型載入、資料庫操作與 RAG 流程更清楚：

| 模組 | 負責內容 |
|------|----------|
| `main.py` | FastAPI 入口、靜態頁面與 API 路由 |
| `db.py` | SQLite 初始化、查詢、寫入與簡易 migration |
| `models.py` | Embedding、Reranker、Whisper 載入與 GPU 記憶體管理 |
| `llm_service.py` | Ollama / DeepSeek provider 設定、文字生成與停止生成 |
| `audio_service.py` | 音檔儲存、轉錄、摘要、檔名建議、來源刪除與資料一致性檢查 |
| `rag_service.py` | 語意切分、ChromaDB 索引、混合檢索、Reranker、問答與引用過濾 |
| `schemas.py` | Pydantic 請求資料模型 |

### 知識建立流程

```mermaid
flowchart LR
    A["錄音 / 上傳音檔"] --> B["FastAPI"]
    B --> C["Whisper 語音辨識"]
    C --> D["摘要 / 深度分析"]
    D --> E["語意切分與摘要片段"]
    E --> F["BGE-M3 向量化"]
    F --> G["ChromaDB 儲存"]
    B --> H["LLM Provider"]
    H --> D

    style A fill:#E3F2FD,stroke:#1565C0,color:#000
    style B fill:#FFF3E0,stroke:#E65100,color:#000
    style C fill:#F3E5F5,stroke:#7B1FA2,color:#000
    style D fill:#FCE4EC,stroke:#C62828,color:#000
    style E fill:#F3E5F5,stroke:#7B1FA2,color:#000
    style F fill:#F3E5F5,stroke:#7B1FA2,color:#000
    style G fill:#E8F5E9,stroke:#2E7D32,color:#000
    style H fill:#F3E5F5,stroke:#7B1FA2,color:#000
```

### RAG 問答流程

```mermaid
flowchart LR
    A["使用者提問"] --> B["BGE-M3 向量化"]
    B --> C["Dense 向量檢索 Top-20"]
    A --> D["jieba 中文分詞"]
    D --> E["BM25 關鍵字檢索 Top-20"]
    C --> F["RRF 排名融合"]
    E --> F
    F --> G["BGE-Reranker 精排至 Top-5"]
    G --> H["LLM Provider 生成回答"]
    H --> I["引用過濾 + 來源追溯"]

    style F fill:#E3F2FD,stroke:#1565C0,color:#000
    style G fill:#FCE4EC,stroke:#C62828,color:#000
    style H fill:#F3E5F5,stroke:#7B1FA2,color:#000
    style I fill:#E8F5E9,stroke:#2E7D32,color:#000
```

---

## 功能特色

### 多筆記本管理
- 建立、重新命名、刪除筆記本
- 每個筆記本擁有獨立的知識庫、對話紀錄與來源管理
- 卡片式首頁導覽，顯示來源數量與最後更新時間

### 音檔上傳與即時錄音
- 支援 MP3 / M4A / WAV / WebM 等常見音訊格式
- 網頁端直接錄音，錄音完成自動上傳處理
- 可選擇自動命名，AI 會依據逐字稿或分析內容產生較容易辨識的來源檔名

### 語音辨識與 AI 摘要
- Whisper `large-v3-turbo` 高速繁體中文語音辨識（VAD + Batch 推理）
- 保留每段語音的時間戳，供引用追溯使用
- 依音檔長度自動選擇快速摘要或長音檔深度分析
- Ollama / DeepSeek 可產生結構化重點、章節摘要、核心主題與推薦問題
- 上傳完成後自動產生推薦問題
- 分析結果也會寫入 ChromaDB，讓問答可以同時參考原文片段與摘要片段

### LLM Provider 切換
- 預設 provider 為本地 Ollama，可透過 `.env` 設定 `VOICERAG_DEFAULT_LLM_PROVIDER`
- 前端可切換 Ollama 或 DeepSeek，切換後會呼叫後端預載對應模型設定
- 支援停止回答：Ollama 會透過 `keep_alive=0` 嘗試釋放模型佔用
- DeepSeek 需要 `DEEPSEEK_API_KEY`；若未設定，前端會顯示不可用狀態

### 四階段混合檢索
1. **Dense 向量檢索**：BGE-M3 語意相似度搜尋
2. **BM25 關鍵字檢索**：jieba 中文分詞，精確匹配人名、專有名詞
3. **RRF 排名融合**：Reciprocal Rank Fusion 合併兩份排名
4. **Reranker 精排**：BGE-Reranker-v2-m3 CrossEncoder 二次排序，取 Top-5

### 智慧問答與來源引用
- 基於 RAG 的知識問答，嚴格依據資料庫內容回答
- 多輪對話記憶（最近 6 則對話上下文）
- 自動過濾不相關引用，只顯示真正支撐答案的來源
- 回答正文自動清理模型自行產生的來源標註

### 逐字稿修正與重新索引
- 查看完整語音逐字稿
- 手動修正 Whisper 辨識錯誤
- 修正後自動重新切分、向量化、重新分析並更新索引
- 可針對單一來源重新產生摘要分析與推薦問題
- 可要求 AI 重新建議來源檔名

### 資料維護
- 資料一致性檢查：自動比對 SQLite / ChromaDB / 音檔三方資料
- 孤兒索引清除：一鍵清理已無對應資料的 ChromaDB segment 目錄

---

## 專案目錄結構

```
rag_project/
├── README.md                        # 專案說明文件
├── AGENTS.md                        # AI 協作指南
├── pyrefly.toml                     # Python 型別檢查設定
├── docs/
│   ├── improvement_plan1.md         # RAG 強化計畫（第一階段）
│   ├── improvement_plan2.md         # 強化計畫（第二階段）
│   ├── notebook_dashboard_plan.md   # 多筆記本功能規劃
│   └── troubleshooting_log.md       # 開發除錯紀錄
├── scripts/                         # 輔助腳本
└── backend/
    ├── .env.example                 # 環境變數範例
    ├── main.py                      # FastAPI 入口，API 路由定義
    ├── db.py                        # SQLite 資料層（筆記本 / 來源 / 對話）
    ├── models.py                    # AI 模型載入（Embedding / Reranker / Whisper）
    ├── llm_service.py               # LLM provider 管理（Ollama / DeepSeek）
    ├── audio_service.py             # 音檔管理（上傳 / 轉錄 / 摘要 / 刪除）
    ├── rag_service.py               # RAG 核心邏輯（檢索 / 排序 / 問答 / 切分）
    ├── schemas.py                   # Pydantic 請求模型
    ├── requirements.txt             # Python 套件清單
    ├── static/
    │   └── index.html               # 前端單頁式介面（1700+ 行）
    ├── audio_uploads/               # 上傳音檔儲存目錄（.gitignore）
    ├── chroma_db/                   # ChromaDB 向量資料（.gitignore）
    ├── notebooks.db                 # SQLite 資料庫（.gitignore）
    └── venv/                        # Python 虛擬環境（.gitignore）
```

---

## 環境需求

| 項目 | 需求 |
|------|------|
| **作業系統** | Windows 10 / 11 (64-bit) |
| **Python** | 3.10+ |
| **GPU** | NVIDIA GPU（建議 RTX 3060 12GB 以上） |
| **VRAM** | 建議 16GB（同時載入 Embedding + Reranker + Whisper + LLM） |
| **RAM** | 建議 32GB |
| **Ollama** | 本地 provider 需預先安裝並拉取模型 `qwen3.5:9b-q4_K_M` |
| **DeepSeek API Key** | 選用 DeepSeek provider 時才需要 |

### VRAM 預估

| 元件 | 估算 VRAM |
|------|-----------|
| BGE-M3 Embedding (FP16) | ~2 GB |
| BGE-Reranker-v2-m3 (FP16) | ~1.5 GB |
| Whisper large-v3-turbo (FP16) | ~3 GB（後端啟動時載入） |
| Ollama `qwen3.5:9b-q4_K_M` | 約 6-9 GB，依 Ollama 與 GPU offload 設定而定 |
| **合計** | **約 12.5-15.5 GB** |

> Whisper 會在後端啟動時載入 GPU；轉錄前系統會主動卸載 Ollama 模型並清理 CUDA 快取，以降低同時載入多個模型造成的 VRAM 壓力。

---

## 安裝與啟動

### 1. 安裝 Ollama 並拉取模型

```bash
# 安裝 Ollama：https://ollama.com/download
ollama pull qwen3.5:9b-q4_K_M
```

### 2. 建立虛擬環境並安裝套件

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> 注意：本專案的 `requirements.txt` 固定使用 CUDA 12.8 版 PyTorch：
> `torch==2.11.0+cu128`、`torchaudio==2.11.0+cu128`、`torchvision==0.26.0+cu128`。
> 這類 `+cu128` 套件不在一般 PyPI 來源中，第一次安裝時請先使用 PyTorch 官方 CUDA 12.8 套件來源安裝 PyTorch，再安裝其餘套件：
>
> ```powershell
> python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchaudio==2.11.0+cu128 torchvision==0.26.0+cu128
> python -m pip install -r requirements.txt
> ```
>
> 如果你的 GPU 或驅動程式不支援 CUDA 12.8，請到 PyTorch 官網選擇符合本機環境的 CUDA 版本，並同步調整 `requirements.txt` 中的 PyTorch 版本。

### 3. 建立環境變數設定檔

在 `backend` 資料夾中，將 `.env.example` 複製為 `.env`。也就是把 `backend/.env.example` 複製成 `backend/.env`。

```powershell
Copy-Item .env.example .env
```

接著開啟 `backend/.env`，確認或填入以下設定：

```env
VOICERAG_DEFAULT_LLM_PROVIDER=ollama
OLLAMA_LLM_MODEL=qwen3.5:9b-q4_K_M
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=填入你的API_KEY
```

若只使用本地 Ollama，可以先不填 DeepSeek API Key；若要使用 DeepSeek，請把 `填入你的API_KEY` 換成自己的 DeepSeek API Key。使用 DeepSeek 時，摘要、分析或問答需要的文字內容會送到 DeepSeek API。

### 4. 啟動服務

```bash
cd backend
venv\Scripts\activate
uvicorn main:app --reload
```

服務啟動後，開啟瀏覽器前往 `http://localhost:8000` 即可使用。

> 首次啟動會自動下載 AI 模型（BGE-M3、BGE-Reranker、Whisper），需要穩定網路連線。

---

## API 一覽

### LLM Provider

| 方法 | 路徑 | 說明 |
|------|------|------|
| `GET` | `/api/llm-providers/` | 取得可用 provider、預設 provider 與模型設定 |
| `POST` | `/api/llm-providers/preload` | 預載指定 provider 需要的本地模型設定 |

### 筆記本管理

| 方法 | 路徑 | 說明 |
|------|------|------|
| `GET` | `/api/notebooks/` | 取得所有筆記本列表 |
| `POST` | `/api/notebooks/` | 建立新筆記本 |
| `PUT` | `/api/notebooks/{id}` | 修改筆記本名稱 |
| `GET` | `/api/notebooks/{id}` | 取得筆記本詳細資料（來源、對話、推薦問題） |
| `DELETE` | `/api/notebooks/{id}` | 刪除筆記本及所有關聯資料 |

### 來源管理

| 方法 | 路徑 | 說明 |
|------|------|------|
| `POST` | `/upload-audio/` | 上傳音檔（自動轉錄 + 摘要 / 分析 + 向量化，可帶 `llm_provider`） |
| `GET` | `/api/notebooks/{id}/sources/{sid}/transcript` | 取得來源逐字稿 |
| `PUT` | `/api/notebooks/{id}/sources/{sid}/transcript` | 修正逐字稿、重新分析並重新索引 |
| `GET` | `/api/notebooks/{id}/sources/{sid}/audio` | 下載來源音檔 |
| `PUT` | `/api/notebooks/{id}/sources/{sid}/filename` | 修改來源檔名 |
| `POST` | `/api/notebooks/{id}/sources/{sid}/filename/suggest` | 依逐字稿或分析內容產生 AI 建議檔名 |
| `POST` | `/api/notebooks/{id}/sources/{sid}/reanalyze` | 重新產生來源摘要分析、摘要片段與推薦問題 |
| `DELETE` | `/api/notebooks/{id}/sources/{sid}` | 刪除來源（同步刪除向量 + 音檔） |

### 問答

| 方法 | 路徑 | 說明 |
|------|------|------|
| `POST` | `/ask-question/` | RAG 知識庫問答，可帶 `llm_provider` |
| `POST` | `/api/stop-answer/` | 停止目前 provider 的回答生成；Ollama 會嘗試卸載模型 |
| `POST` | `/api/notebooks/{id}/suggested-questions/{qid}/used` | 標記推薦問題為已使用 |

### 系統維護

| 方法 | 路徑 | 說明 |
|------|------|------|
| `GET` | `/api/data-consistency` | 資料一致性檢查 |
| `POST` | `/api/maintenance/clear-orphan-chroma` | 清除孤兒 Chroma 索引 |

---

## 資料庫結構

### SQLite（`notebooks.db`）

```mermaid
%%{
  init: {
    "theme": "base",
    "themeVariables": {
      "background": "#000000",
      "primaryColor": "#000000",
      "primaryTextColor": "#ffffff",
      "primaryBorderColor": "#ffffff",
      "lineColor": "#ffffff",
      "textColor": "#ffffff",
      "entityBkg": "#000000",
      "entityTextColor": "#ffffff",
      "attributeBackgroundColorOdd": "#000000",
      "attributeBackgroundColorEven": "#000000",
      "attributeTextColor": "#ffffff"
    },
    "themeCSS": "
      .er.entityBox {
        fill: #000000 !important;
        stroke: #ffffff !important;
      }

      .er.attributeBoxOdd,
      .er.attributeBoxEven {
        fill: #000000 !important;
        stroke: #ffffff !important;
      }

      .er.entityLabel,
      .er.attributeLabel,
      .er.relationshipLabel {
        fill: #ffffff !important;
        color: #ffffff !important;
      }

      .er.relationshipLine {
        stroke: #ffffff !important;
      }
    "
  }
}%%

erDiagram
    notebooks {
        TEXT id PK "UUID"
        TEXT name "筆記本名稱"
        TEXT icon "Emoji 圖示"
        TEXT updated_at "最後更新時間"
    }

    sources {
        TEXT id PK "UUID"
        TEXT notebook_id FK "所屬筆記本"
        TEXT filename "音檔名稱"
        TEXT added_at "上傳時間"
        TEXT transcript_text "完整逐字稿"
        TEXT timed_segments "時間戳 JSON"
        TEXT transcript_updated_at "逐字稿修改時間"
        TEXT indexed_at "最後索引時間"
        TEXT analysis_mode "quick / deep"
        TEXT analysis_status "pending / completed / failed"
        TEXT analysis_json "摘要分析 JSON"
        TEXT analysis_updated_at "分析更新時間"
    }

    messages {
        INTEGER id PK "自增"
        TEXT notebook_id FK "所屬筆記本"
        TEXT sender "User / AI"
        TEXT text "訊息內容"
        TEXT created_at "建立時間"
        TEXT references_json "引用來源 JSON"
    }

    suggested_questions {
        INTEGER id PK "自增"
        TEXT notebook_id FK "所屬筆記本"
        TEXT source_filename "來源檔名"
        TEXT question "推薦問題"
        INTEGER priority "優先序"
        INTEGER used "是否已使用"
        TEXT created_at "建立時間"
    }

    notebooks ||--o{ sources : "擁有"
    notebooks ||--o{ messages : "擁有"
    notebooks ||--o{ suggested_questions : "擁有"
```

### ChromaDB

每個筆記本對應一個獨立 Collection（`notebook_{id}`）。目前會索引兩類內容：

- **原文片段**：逐字稿經語意切分後的片段，供精準引用與時間戳追溯使用。
- **分析片段**：來源的全局摘要與章節摘要，供整體性問題與長音檔問答使用。

每個 ChromaDB 文件包含：

| 欄位 | 說明 |
|------|------|
| `id` | 原文片段為 `{source_id}_chunk_{index}_{隨機碼}`；摘要片段為 `{source_id}_global_summary_{隨機碼}` 或 `{source_id}_chapter_{index}_{隨機碼}` |
| `embedding` | 1024 維向量（BGE-M3 FP16） |
| `document` | 知識片段文字 |
| `metadata.source` | 來源檔名 |
| `metadata.source_id` | 來源 UUID |
| `metadata.doc_type` | `transcript_chunk`、`global_summary` 或 `chapter_summary` |
| `metadata.chunk_index` | 片段序號 |
| `metadata.chapter_index` | 章節摘要序號；原文片段通常不使用 |
| `metadata.char_start` / `char_end` | 在逐字稿中的字元位置 |
| `metadata.start_time` / `end_time` | 對應音檔的時間範圍（秒） |
| `metadata.time_range` | 格式化時間文字（如 `02:13-02:48`） |
| `metadata.analysis_version` | 摘要分析版本；只存在於分析片段 |

---

## 已知限制與注意事項

- **GPU 必要**：Whisper、Embedding、Reranker 均依賴 CUDA GPU，無 GPU 環境無法正常運行
- **模型載入順序**：`sentence_transformers` 必須先於 `faster_whisper` 載入，否則 Windows 下會因 DLL / OpenMP 衝突導致無聲崩潰（詳見 [troubleshooting_log.md](docs/troubleshooting_log.md)）
- **Ollama provider 需要本機服務**：使用本地 provider 前，需確認 Ollama 服務可用且已拉取 `qwen3.5:9b-q4_K_M` 模型
- **DeepSeek provider 不是本地推理**：使用 DeepSeek 時，相關文字內容會送往 DeepSeek API，需要自行評估資料隱私與 API 配額
- **VRAM 管理**：系統會在轉錄前主動卸載 Ollama 模型以釋放 VRAM，避免 OOM 崩潰
- **單人使用設計**：目前未實作使用者認證與多人並行機制

---

## 技術亮點

### 語意切分（Semantic Chunking）
**讓每個片段保留完整語意。** 文字超過 200 字時，系統使用 BGE-M3 計算相鄰句子的語意相似度，在語意斷裂點切分，避免只用固定長度切割造成上下文破碎。

### 四階段混合檢索 + 引用過濾
**提升找得到與找得準的機率。** 系統結合 Dense 向量檢索、BM25 關鍵字檢索、RRF 融合與 Reranker 精排，從 20 份候選中篩選至 Top-5。回答後再過濾引用，只保留真正支撐答案的來源。

### 長音檔深度分析
**把長逐字稿整理成可問答的知識結構。** 系統會依照逐字稿長度與音檔時間判斷是否啟用深度分析，長音檔會拆成章節後分段分析，再產生全局摘要、章節摘要、核心主題與推薦問題。

### 原文與摘要雙索引
**同時支援細節追問與整體理解。** 原始逐字稿片段保留時間戳，適合回答「哪裡提到」這類問題；摘要片段則保存全局與章節重點，適合回答「整段內容重點是什麼」這類問題。

### GPU 記憶體動態管理
**降低多模型同時執行的 VRAM 壓力。** Whisper、Embedding 與 Reranker 會在後端啟動時載入 GPU。轉錄音檔前，系統會呼叫 Ollama API 卸載 LLM 模型（`keep_alive=0`），並清空 CUDA 快取、觸發 GC，降低 OOM 風險。

### 模型正文清理
**讓回答正文與引用區塊分工清楚。** AI 回答後，系統會自動移除模型自行產生的「參考來源」、「資料來源」等文字，引用統一由前端控制顯示，避免重複或格式混亂。

---

## 競品比較

| 功能 | VoiceRAG | NotebookLM | Notion AI | 訊飛聽見 |
|------|----------|------------|-----------|----------|
| 語音轉文字 | ✅ 本地端 GPU | ✅ 雲端 | ❌ | ✅ 雲端 |
| AI 知識結構化 | ✅ | ✅ | ✅ | ⚠️ 有限 |
| RAG 知識問答 | ✅ 混合檢索 + 精排 | ✅ 雲端 | ⚠️ 有限 | ❌ |
| 來源引用追溯 | ✅ 含時間戳 | ✅ | ❌ | ❌ |
| 資料隱私 | ✅ Ollama 模式可本地端；DeepSeek 模式為雲端 | ❌ 雲端 | ❌ 雲端 | ❌ 雲端 |
| 繁體中文 | ✅ 專案優化 | ✅ | ✅ | ⚠️ 以簡體情境較常見 |
| 費用 | 本機硬體成本 | 依服務方案 | 月費制 | 按量或方案計費 |

---

> **VoiceRAG** · 資訊管理學系畢業專題 
