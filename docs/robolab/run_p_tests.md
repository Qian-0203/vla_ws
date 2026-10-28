# RoboLab P-tests 執行說明（接在選項 A 之後）

給已經跑過[選項 A](run_option_a.md) 的機器。環境、模型、交回方式都和選項 A 一樣，只是換一組指令。

## 測什麼

同一個模型 π0.5-DROID，同樣 3 個 RoboLab 原有場景，只換指令，每格 20 集。對應 LIBERO-Spatial 定版文件 §6 的假說：

| 組 | 場景 / 任務檔 | key（每個 key 一個 run） | 假說 | 集數 |
|---|---|---|---|--:|
| `mustard` | two_bin：`LangMustardIn{Left,Right}BinTask` | `p3_neg_not` `p5_contrast` `p5_len_ctrl` `p4_syn_common` `p4_syn_rare` `p4_functional` `p4_zh` `p7_robot_frame` `p7_image_frame` | P3 否定、P4 用字距離、P5 對比句、P7 視角 | 9 × 2 × 20 = 360 |
| `cube` | rubiks_cube_banana_bowl：`LangRubiksCubeOrBananaTask` | `p1_ask_cube` `p1_ask_banana` `p3_not_banana` `p3_not_cube` `p5_cube_not_banana` `p5_banana_not_cube` | P1 換目標、P3 否定、P5 對比句 | 6 × 20 = 120 |
| `size` | butter_raisin_box_grey_bin：`Lang{Larger,Smaller}Object*Task` | `p3_neg` `p5_contrast` `p5_len_ctrl` | P3 否定、P5 對比句 | 3 × 2 × 20 = 120 |

- 合計 600 集。依選項 A 的速度（260 集約 3.7 小時），單卡約 8.5 小時，3 張卡平行約 3 小時。
- 每個 key 的實際句子在 `robolab_lang/tasks/<任務檔>.py` 的 `instruction`。

## 步驟

1. **等選項 A 全部跑完**再更新檔案，以免動到正在跑的 run。

   ```bash
   cd vla_ws && git pull origin robolab/option-a-runkit
   rm -rf ~/robolab_ws/robolab_lang && cp -r robolab_lang ~/robolab_ws/robolab_lang
   ```

2. policy server 照選項 A 步驟 2 啟動（已經在跑就不用重開）。要用多張卡平行跑時，每張卡各開一個 server，port 用 8765、8766、8767。

3. 執行，一組一張卡（三個指令可以同時在三個 tmux 視窗跑）：

   ```bash
   cd ~/robolab_ws
   GROUP=mustard GPU=0 PORT=8765 HOST_TAG=<機器名> bash robolab_lang/run_p_tests.sh
   GROUP=cube    GPU=1 PORT=8766 HOST_TAG=<機器名> bash robolab_lang/run_p_tests.sh
   GROUP=size    GPU=2 PORT=8767 HOST_TAG=<機器名> bash robolab_lang/run_p_tests.sh
   ```

   - 只有一張卡：`GROUP=all`，三組依序跑。
   - `NUM_ENVS` / `NUM_RUNS` 和選項 A 一樣，要保持 `NUM_ENVS × NUM_RUNS = 20`。
   - **選項 A 如果是用自己的 Apptainer 腳本跑的**：用同一個腳本，對上表每一組設定 `STAGE=p-<組名>`、`TASKS=<任務檔>`、`KEYS=<key 清單>` 執行即可。`run_p_tests.sh` 本身只做這件事。

4. run_id 是 `<RUN_PREFIX>_robolab_pi05_p-<組名>_<機器名>_<key>`，輸出格式和選項 A 相同。

## 交回結果

照[選項 A 的 PR 流程](run_option_a.md#交回結果用-pr給執行的-agent)，但另開一個 branch 和 draft PR：

- branch：`handoff/<機器名>-p-tests`，base 一樣是 `origin/results-log/robolab`
- PR 標題：`RoboLab P-tests results: <機器名>`
- 每個 run 跑完就 push；全部跑完後上傳 tgz、貼連結，再 `gh pr ready`
