# VoiceRAG 多筆記本導覽功能 (Notebook Dashboard) 實作計畫書

為擴充現有系統，將在使用者進入「知識助理」對話介面前，新增一個**「筆記本選擇導覽首頁」**（類似 Google NotebookLM 的設計）。這意味著系統架構將從原本的「單一知識庫」升級為「多知識庫管理系統」。

以下為此功能的詳細架構與實作規劃：

---

## 一、 功能核心需求

1. **進入點改變**：使用者開啟網頁時，預設顯示「最近開啟的筆記本」列表。
2. **多空間隔離**：每個「筆記本」視為一個獨立的知識空間。在 A 筆記本上傳的音檔與進行的問答，完全獨立於 B 筆記本。
3. **卡片式 UI 設計**：
   - **新增卡片**：固定於左側，點擊即可「建立新的筆記本」。
   - **歷史卡片**：顯示已建立的筆記本，包含圖示、標題、最後開啟日期，以及包含的「來源數量」。點擊後正式進入該筆記本的專屬知識助理工作區。
   - **卡片管理選單**：歷史卡片右上角設有「三個點」選項按鈕，點擊展開浮動選單，提供「編輯標題」與「刪除」功能。

---

## 二、 前端架構調整 (Frontend: `static/index.html`)

不需建立新的 HTML 檔案，而是透過 JavaScript 控制同一個單頁面應用程式 (SPA) 的**視圖切換 (View Routing)**。

### 1. 視圖狀態管理
將畫面分為兩個主要區塊，預設只顯示「首頁」，點擊筆記本後切換至「工作區」。
- `div#dashboard-view` (新增)：包含「建立新的筆記本」按鈕與歷史筆記本卡片列表。
- `div#workspace-view` (現有介面)：原有的左側來源欄與右側對話框，預設加上 `hidden` 隱藏。

### 2. 筆記本首頁 UI 實作
- 採用全黑深色背景 (`bg-[#131314]`)。
- 實作響應式網格 (CSS Grid) 呈現卡片。
- 卡片設計採用深灰底色 (`bg-[#1e1f22]`)、圓角、Hover 浮動特效。

### 3. API 串接邏輯
- 網頁載入時，發送 `GET /api/notebooks` 取得筆記本清單並渲染卡片。
- 點擊「建立新筆記本」時，發出請求建立新空間，然後切換至 `workspace-view`。
- 進入 `workspace-view` 時，前端需在記憶體中保存當前的 `notebook_id`。後續上傳音檔或詢問問題時，皆須將 `notebook_id` 一併傳遞給後端 API。

---

## 三、 後端架構調整 (Backend: `main.py`)

為了支援「多筆記本」，後端必須進行資料隔離的架構重構。

### 1. 筆記本元資料管理 (Metadata Storage)
由於 FastAPI 重新啟動後記憶體變數會清除，需要持久化機制來記錄筆記本清單與基本資訊。
- **方案**：使用 Python 內建的 `sqlite3` 或基礎 JSON 檔案儲存。
- **資料結構**：
  - `notebook_id` (唯一識別碼, UUID)
  - `name` (筆記本名稱)
  - `icon` (Emoji 圖示)
  - `updated_at` (最後更新時間)
  - `sources` (該筆記本擁有的音檔來源清單)

### 2. ChromaDB 向量庫分區 (Collections)
原架構將所有資料存入單一 `mis_knowledge` collection。
- **改動**：每個筆記本對應 ChromaDB 中的一個**獨立 Collection**。例如：`collection_name = f"notebook_{notebook_id}"`。
- **目的**：確保 RAG 檢索時具備絕對的硬隔離，不會跨筆記本檢索到不相干的資料。

### 3. 新增與修改 API 端點
- 新增 `GET /api/notebooks`：回傳所有筆記本的清單與來源數量。
- 新增 `POST /api/notebooks`：建立一個新的筆記本（自動產生 ID 與初始名稱）。
- 新增 `PUT /api/notebooks/{notebook_id}`：更新特定筆記本的資訊（例如：修改標題）。
- 新增 `DELETE /api/notebooks/{notebook_id}`：刪除特定筆記本，並同步清除關聯的 ChromaDB Collection 與資料庫紀錄。
- 修改 `POST /upload-audio/`：接收 `notebook_id`，將音檔存入對應的資料夾與指定的 ChromaDB Collection，並更新該筆記本的來源數量記錄。
- 修改 `POST /ask-question/`：接收 `notebook_id`，從指定的 ChromaDB Collection 中檢索知識並進行回答。

---

## 四、 開發實作階段

為確保系統穩定，建議分為兩個階段進行實作：

- **階段一（後端重構與 API 建立）**：建立 SQLite 記錄檔與動態切換 ChromaDB Collection 的底層邏輯，並完成 Notebook 的 CRUD API。
- **階段二（前端 UI 與 API 串接）**：實作首頁卡片介面，並將現有的上傳與問答功能綁定到動態的 `notebook_id` 狀態上。
