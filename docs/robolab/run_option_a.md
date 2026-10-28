# RoboLab 語言試跑（選項 A）執行說明

給協助跑實驗的人。照著做就能從一台空的 GPU 機器跑完，結果打包交回。設計細節見 [`plan.md`](plan.md) §4–5。

## 這次要跑什麼

- **模型**：π0.5-DROID（`pi05_droid_jointpos`），也就是 RoboLab 排行榜上的 π0.5。
- **模擬器**：Isaac Sim 5.0 / Isaac Lab 2.2.0，RoboLab commit `b6cfcf6`，和排行榜相同。
- **任務**：4 個 RoboLab 原有場景、13 個任務。場景和成功判定都是官方的，只換指令。
- **指令**：每個任務 3 種，用 `--instruction-type` 選擇：

| key | 內容 |
|---|---|
| `default` | 官方原句 |
| `blank` | `"."`（RoboLab 不接受空字串） |
| `no_target` | 拿掉決定目標的詞，例如 "Put the mustard in the bin" |

- **集數**：13 任務 × 3 指令 × 20 集 = **780 集**。在 RTX PRO 6000（96 GB）上開 10 個平行環境，約 3–6 GPU 小時。

| 場景 | 任務檔（`robolab_lang/tasks/`） | 測的語言 |
|---|---|---|
| two_bin | `LangMustardInLeftBinTask`, `LangMustardInRightBinTask` | 左 / 右 |
| bowls_2_table | `LangBowlStackingLeftOnRightTask`, `LangBowlStackingRightOnLeftTask` | 左 / 右（在相機裡是上下） |
| butter_raisin_box_grey_bin | `LangLargerObjectRaisinBoxInBinTask`, `LangSmallerObjectButterInBinTask` | 大 / 小 |
| rubiks_cube_banana_bowl | `LangRubiksCube{LeftOf,Behind,InFrontOf}BowlTask`, `LangRubiksCube{And,Or,Then}BananaTask`, `LangBananaThenRubiksCubeTask` | 方位、和 / 或 / 然後 |

## 所需檔案

全部在這個 repo 的 `robolab_lang/`（branch `robolab/option-a-runkit`）：

| 檔案 | 用途 |
|---|---|
| `setup_robolab.sh` | 一次性安裝：clone RoboLab 和 openpi、build 兩個 docker image |
| `docker/Dockerfile.ffmpeg` | 在 RoboLab image 上加 ffmpeg（存影片用） |
| `serve_pi05.sh` | 啟動 π0.5 policy server |
| `run_option_a.sh` | 跑選項 A：每種指令一個 run，寫好紀錄檔 |
| `run_robolab.sh` | 在 RoboLab container 裡跑一個 eval（被 `run_option_a.sh` 呼叫） |
| `run_pi0.py` | RoboLab 官方 π0 client 加上「從檔案載入任務」，評估流程和官方相同 |
| `tasks/*.py` | 13 個任務檔：繼承官方任務，只多加指令 key |
| `make_run_record.py`, `outcomes.py` | 跑完後產生 `episodes.jsonl`、`summary.csv`，並記錄模型實際動了哪個物體 |

外部來源（`setup_robolab.sh` 會自動抓）：

- `github.com/NVlabs/RoboLab` @ `b6cfcf6`，含 git-lfs 場景資產，約 6 GB
- `nvcr.io/nvidia/isaac-lab:2.2.0`（docker image，數十 GB）
- `github.com/xuningy/openpi` @ `aa64205`（policy server）
- checkpoint `gs://openpi-assets-simeval/pi05_droid_jointpos`：公開，約 12 GB，第一次啟動 server 時自動下載

## 機器需求

- NVIDIA **RTX** GPU（要有 RT core，Isaac Sim 需要；A100/H100 不行）
  - 建議 ≥ 48 GB
  - 24 GB 可以跑，但平行環境要調少
  - 16 GB 沒試過
- Ubuntu、docker、nvidia-container-toolkit（`docker run --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` 要能跑）
- git、git-lfs、[uv](https://docs.astral.sh/uv/)
- 硬碟剩 60 GB 以上

## 步驟

### 1. 取得檔案並安裝（1–2 小時，大多是下載）

```bash
git clone -b robolab/option-a-runkit https://github.com/Qian-0203/vla_ws.git
cd vla_ws
bash robolab_lang/setup_robolab.sh          # 裝到 ~/robolab_ws，可用 WS=... 改位置
```

如果 `docker pull nvcr.io/...` 被拒，先 `docker login nvcr.io`。帳號填 `$oauthtoken`，密碼填 NGC API key。

### 2. 啟動 policy server（另開一個 terminal 或 tmux）

```bash
cd ~/robolab_ws
GPU=0 PORT=8765 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 bash robolab_lang/serve_pi05.sh
```

- `XLA_PYTHON_CLIENT_MEM_FRACTION` 是 JAX 預先佔用的顯存比例，server 實際約需 8–10 GB。依卡的大小調整：
  - 96 GB 卡：0.2
  - 48 GB 卡：0.25
  - 24 GB 卡：0.45
- 第一次啟動會下載約 12 GB 的 checkpoint。看到 server 開始 listen 就可以進下一步。

### 3. Smoke test（約 5–10 分鐘）

第一次啟動 Isaac Sim 要編譯 shader，會比較久。

```bash
cd ~/robolab_ws
TASKS=LangMustardInLeftBinTask KEYS=default NUM_ENVS=1 NUM_RUNS=1 STAGE=smoke \
    GPU=0 PORT=8765 HOST_TAG=<機器名> bash robolab_lang/run_option_a.sh
```

最後會印出 `summary.csv`，有一列 `n=1` 就代表整條流程是通的。

### 4. 正式跑

```bash
cd ~/robolab_ws
GPU=0 PORT=8765 HOST_TAG=<機器名> NUM_ENVS=10 NUM_RUNS=2 bash robolab_lang/run_option_a.sh
```

- 跑法：三種指令依序各跑一個 run，每個 run 跑 13 個任務 × 20 集。
- 調整 `NUM_ENVS` 時要保持 `NUM_ENVS × NUM_RUNS = 20`。小卡用 `NUM_ENVS=5 NUM_RUNS=4`；如果顯存不夠（OOM），改 `NUM_ENVS=4 NUM_RUNS=5`。
- 中途中斷了，用同一個 `RUN_PREFIX=<日期-時間>` 重跑，會接著跑。前綴印在每個 run 開頭的 log 裡。
- 時間不夠時，可以先跑一半：`KEYS="default no_target" NUM_ENVS=10 NUM_RUNS=1`，即 13 × 2 × 10 = 260 集。

### 5. 交回結果

每個 run 會在 `~/robolab_ws/runs/<run_id>/` 底下產生：

- `run.json`：設定，跑之前就寫好
- `episodes.jsonl`：每集一列
- `summary.csv`：任務 × 指令的成功率
- `SHA256SUMS`
- `robolab_output/`：RoboLab 原始輸出，含 HDF5 和影片

把整個 `runs/` 打包交回，連同 `~/robolab_ws/logs/`：

```bash
cd ~/robolab_ws && tar czf robolab_option_a_runs.tgz runs logs
```

如果檔案太大，至少交回每個 run 的 `run.json`、`episodes.jsonl`、`summary.csv`、`SHA256SUMS`，以及 `robolab_output/episode_results.jsonl`。

## 常見問題

- **`ImportError` 或 transformers 版本錯**：container 一律帶 `-e PYTHONNOUSERSITE=1`（腳本已經加了）。
- **影片沒存下來**：沒有用 `robolab:b6cfcf6-ffmpeg` 這個 image。
- **Isaac Sim 卡在啟動**：第一次啟動會編譯 shader，可能要 5–10 分鐘。cache 存在 `~/robolab_ws/cache/`，之後就會快。
- **終端出現 `cuDeviceGetUuid` 警告**：無害，可以忽略。
- **`no policy server on port 8765`**：步驟 2 的 server 還沒起來，或 port 號不一樣。
