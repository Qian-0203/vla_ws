# RoboLab：論文測試的擴充（接在選項 A 之後）

給已經跑過[選項 A](run_option_a.md) 的機器。環境、模型、交回方式都和選項 A 相同，只換指令。

目的：把 ARR 稿（LIBERO-Spatial，9 個 policy）的測試搬到 RoboLab 原有場景，用 π0.5-DROID 檢查論文的發現在換了模型和場景後是否還成立。

## 和論文的對應

| 論文條件 | 論文發現 | 這裡的 key | 場景（組） |
|---|---|---|---|
| switch | 要求另一個物體時常送回原本的 | `switch_cube` / `switch_banana` | cube |
| negation | 常送出被排除的物體 | `negation_not_banana` / `negation_not_cube`；`negation`（「not the butter」，和習慣相反） | cube、size |
| negation（目的地） | — | `negation`（「the bin that is not on the right」） | mustard |
| no target | 沒指定時偏向原本的 | `no_target`（在選項 A 裡） | 全部 |
| paraphrase | 幾乎不扣分 | `paraphrase_common` / `_rare` / `_functional` / `_zh`（比論文多一段：罕見字、功能描述、中文） | mustard |
| irrelevant content | 扣分小 | `irrelevant_content`（「…, and let go gently」） | mustard、size |
| distractor mention | 扣 21–46 分 | `distractor_mention`；`distractor_cube` / `distractor_banana` | mustard、size、cube |
| spatial wording | 扣 34–54 分 | `spatial_robot`（「on the robot's left」）/ `spatial_image`（「on the left side of the image」） | mustard |
| third-object control | 場景和用字傷到不同模型 | **沒有安排**（見缺口） | — |

每個 key 的句子在 `robolab_lang/tasks/<任務檔>.py` 的 `instruction`。

## 規模

| 組 | 任務檔 | key 數 | 集數 |
|---|---|--:|--:|
| `mustard` | `LangMustardIn{Left,Right}BinTask` | 9 | 9 × 2 × 20 = 360 |
| `cube` | `LangRubiksCubeOrBananaTask` | 6 | 6 × 20 = 120 |
| `size` | `Lang{Larger,Smaller}Object*Task` | 3 | 3 × 2 × 20 = 120 |

合計 600 集。依選項 A 的速度，單卡約 8.5 小時，3 張卡平行約 3 小時。

## 步驟

1. **等選項 A 全部跑完**再更新檔案：

   ```bash
   cd vla_ws && git pull origin robolab/option-a-runkit
   rm -rf ~/robolab_ws/robolab_lang && cp -r robolab_lang ~/robolab_ws/robolab_lang
   ```

2. policy server 照選項 A 步驟 2 啟動。多卡平行時每張卡各開一個，port 8765、8766、8767。

3. **先跑 smoke test**，確認新 key（含中文句）能跑：

   ```bash
   GROUP=mustard ONLY_KEYS=paraphrase_zh NUM_ENVS=1 NUM_RUNS=1 RUN_PREFIX=smoke GPU=0 PORT=8765 \
       HOST_TAG=<機器名> bash robolab_lang/run_paper_tests.sh
   ```

   - 印出的 `summary.csv` 有 `n=1`，最後兩欄（`requested_delivered`、`other_delivered`）有數字，就代表流程通了。

4. 正式跑，一組一張卡：

   ```bash
   GROUP=mustard GPU=0 PORT=8765 HOST_TAG=<機器名> bash robolab_lang/run_paper_tests.sh
   GROUP=cube    GPU=1 PORT=8766 HOST_TAG=<機器名> bash robolab_lang/run_paper_tests.sh
   GROUP=size    GPU=2 PORT=8767 HOST_TAG=<機器名> bash robolab_lang/run_paper_tests.sh
   ```

   - 只有一張卡時用 `GROUP=all`。
   - 選項 A 如果是用自己的 Apptainer 腳本：對上表每組設定 `STAGE=ext-<組名>`、`TASKS`、`KEYS` 跑同一個腳本，並**把那支腳本一起放進 PR**。

5. run_id 是 `<RUN_PREFIX>_robolab_pi05_ext-<組名>_<機器名>_<key>`。

## 結果欄位

`summary.csv` 新增兩欄，對應論文的違規計數：

- **`requested_delivered`**：要求的目標進了容器，另一個沒有。
- **`other_delivered`**：另一個物體或容器被送達。
  - negation：被排除的那一個
  - switch：沒被要求的那一個
  - mustard：另一個 bin

`cube` 組的官方 `success` 是「任一物體進碗」，**不能當成功率用**；請看 `requested_delivered`。

## 交回結果

照[選項 A 的 PR 流程](run_option_a.md#交回結果用-pr給執行的-agent)，另開 branch `handoff/<機器名>-paper-tests` 和 draft PR，標題 `RoboLab paper-test results: <機器名>`。

## 已知缺口（2026-10-29 檢查）

| # | 缺口 | 影響 | 狀態 |
|---|---|---|---|
| G1 | **每集的物體起始位置完全相同。** RoboLab 每次 reset 都回到場景檔的固定位置（已用 gate 的 50 集 HDF5 確認）。一格 20 集等於同一個擺放重複 20 次 | 信賴區間低估了不確定性。「總是先拿奶油」可能只是這個擺放下奶油的位置，不一定是物體本身。論文每個任務有 50 個起始狀態 | **未處理**。建議加「兩物體交換位置」的對照，再加小幅隨機位移 |
| G2 | 機器是 H200 + Apptainer，不是 RTX + docker；那台機器沒有正式跑過 gate | 和排行榜環境可能有差。目前只有 RubiksCubeAndBanana 一個數字對得上 | Apptainer 腳本要放進 repo；在那台機器補跑 gate |
| G3 | 新 key 和新欄位只用舊資料離線驗證過，沒有實跑 | 可能在跑的時候才出錯，例如中文句 | 步驟 3 的 smoke test |
| G4 | 還沒有產生論文表格的分析腳本（對 default 的差值、違規計數） | 結果回來後沒辦法直接比較 | 資料回來前補上 |
| G5 | 和論文不完全同義。論文的 Δ 是對 LIBERO 分數，這裡是對同任務的 default。論文的 switch 句是訓練看過的，這裡不是。mustard 的 negation 是目的地，不是物體 | 寫論文時要說明 | 寫進方法段 |
| G6 | 第三物體的場景控制（C6）沒有安排；和 LIBERO 分數的相關（C2）、排名改變（C4）需要多個模型 | 這三項驗證不到 | 待決定 |
| G7 | 每格 20 集，論文每個條件 500 集 | 只能當篩選 | 有訊號的格子再加 |
| G8 | 選項 A 的 PR #9 還沒 merge；`docs/robolab/results.md`、eval_log、HF 都還沒更新；tgz 連結還沒給 | 結果還沒正式入庫 | 選項 A 跑完後處理 |
