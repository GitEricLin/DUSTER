# DUSTER v2：無 leakage 的實作與依時間切分的 pipeline

這個版本取代 Google Drive 上 `DUSTER_Code/DUSTER` 與 `DUSTER_Code/DUSTER_A` 兩份程式。兩個 model 現在由同一支 `main.py` 執行，用 `--aspect_mode` 切換：

| 原本的 model | v2 指令 | 評估 protocol |
|---|---|---|
| DUSTER | `--aspect_mode gold` | feature-given：test 時餵入 ground-truth aspect，只當作 upper bound |
| DUSTER_A | `--aspect_mode pred` | decision-time：aspect 只從 history 預測 |
| 不放 aspect token | `--aspect_mode none` | ablation 用 |

使用前，把原專案的 `bleu.py` 與 `rouge.py` 複製到本資料夾。這兩支檔案沒有改動。

## 1. 修正了什麼

| # | 原本的問題 | v2 的修正 |
|---|---|---|
| 1 | `group_data()` 用整個 `reviews.pickle`（train、valid、test 全部）建立 history，`[-100:]` 也沒有排除 target 本身，導致 test sample 的正確答案就在 input 裡 | history 改成**逐筆 target 建立**：一律排除 target 本身；只看得到 split rank 不高於 target 的 record（train sample 只看 train，valid 看 train+valid，test 看全部）；有 timestamp 時只取 `time < t` 的 review，依時間排序後保留最近 `his_len` 篇 |
| 2 | 資料沒有時間欄位，「most recent」實際上是檔案順序 | 用 `prep/add_timestamps.py` 從原始資料補回時間，再用 `prep/make_temporal_split.py` 做 global temporal split |
| 3 | DUSTER 的 GRU 輸出算完就丟掉 | GRU 輸出經 residual 後接 attention pooling；history 以 `pack_padded_sequence` 依實際長度處理 |
| 4 | Attention 把 padding review 也算進去 | Attention 加上 padding mask；history 為空時輸出 0 |
| 5 | Item ID embedding 誤用 `user_embeddings(item)`；user 與 item 共用同一組 `W1/b1` | 改用 `item_embeddings`；`user_fuse` 與 `item_fuse` 分開 |
| 6 | DUSTER_A 把 **user** history 送進 item encoder | user 與 item 各自有獨立的 `HistoryEncoder` |
| 7 | DUSTER_A 的 FMR 拿 `"feature adjective"` 字串跟單一 feature word 比對，結果永遠是 0.00 | FMR 改為和 ground-truth **feature word** 比對 |
| 8 | 「learnable mask」名稱與論文描述不符 | 程式改名為 `feature_gate`：逐維度縮放的向量 g ∈ R^d，所有 review 共用，並非 review-level gate |
| 9 | DUSTER_A 的 aspect 預測涵蓋整個 vocabulary | aspect logits 限制在 feature-word vocabulary 內；可用 `--aspect_source user\|item\|joint` 選擇 |
| 10 | `reviews.pickle` 被重複讀取 3 次；model 在 `forward()` 裡直接讀 corpus | pickle 只讀一次；history 在 model 外部建好再傳入 |
| 11 | 沒有任何 leakage 檢查 | 訓練前自動執行 `check_leakage`，抽樣檢查 valid 與 test，有問題就中止 |

## 2. 補回 timestamp

建議在 Colab 掛載 Google Drive 後執行。

| Dataset | 原始資料 | 連接欄位 | 時間欄位 |
|---|---|---|---|
| Amazon Clothing / Movies | Amazon 2014（McAuley）的 `reviews_Clothing_Shoes_and_Jewelry_5.json.gz` 或 `reviews_Movies_and_TV_5.json.gz`（5-core 或完整版都可以），下載頁：http://jmcauley.ucsd.edu/data/amazon/ | `(reviewerID, asin)` | `unixReviewTime` |
| Yelp | Yelp Open Dataset 的 `yelp_academic_dataset_review.json`，需到 Yelp 官網同意授權後下載 | `(user_id, business_id)` | `date` |
| TripAdvisor | 已在 `TripAdvisor_Dataset/OriginalReviews.json` | `(userID, hotelID)` | `reviewDate` |

範例（Clothing）：

```bash
cd prep
python add_timestamps.py --dataset amazon \
  --pickle /content/drive/MyDrive/Amazon_Datasets/ClothingShoesAndJewelry/reviews.pickle \
  --raw reviews_Clothing_Shoes_and_Jewelry_5.json.gz \
  --out /content/drive/MyDrive/Amazon_Datasets/ClothingShoesAndJewelry/reviews_time.pickle

python make_temporal_split.py \
  --pickle /content/drive/MyDrive/Amazon_Datasets/ClothingShoesAndJewelry/reviews_time.pickle \
  --out_dir /content/drive/MyDrive/Amazon_Datasets/ClothingShoesAndJewelry/temporal \
  --ratio 0.8 0.1 0.1 --drop_cold
```

TripAdvisor 的 `OriginalReviews.json` 約 590 MB。建議先 `pip install ijson`，以 streaming 方式讀取。

每個 dataset 會輸出 `reviews_time.pickle.stats.json`（match rate、時間範圍）與 `temporal/split_stats.json`，這兩份數字要寫進論文。

## 3. 量化原本程式的 leakage（給論文或 response letter 用）

```bash
python prep/audit_old_leakage.py --pickle .../reviews.pickle --index_dir .../1
```

會印出 test sample 的 target 句子出現在原版 user／item history 中的比例。在合成資料上的結果是 100%。

## 4. 訓練與評估

```bash
python main.py --data_path .../reviews_time.pickle --index_dir .../temporal \
               --aspect_mode pred --cuda --seed 1 --checkpoint runs/clothing_pred_s1/
bash run_all.sh                              # 4 個 dataset × {gold, pred} × 5 個 seed
python aggregate_results.py "runs/clothing_pred_s*"
```

- Ablation 參數：`--no_user_hist`、`--no_item_hist`、`--no_gate`、`--no_gru`、`--no_attn`、`--no_duster_mask`、`--aspect_mode none`
- 若沿用原本的 random split（沒有 timestamp）：`--history_scope train`。history 只用 train 的 review，並排除 target，可與 PETER／EPER 的原始設定比較。
- 若想讓 vocabulary 也只由 train 建立：`--vocab_scope train`

**Baseline 也要重跑：** PETER+、EPER、PESI 都要在新的 temporal split 上重跑。DGER 與 SERMON 原論文的數字不能直接比較，只能另外標註。

## 5. 已用合成資料測試

在 CPU（PyTorch 2.x）上用合成資料跑過以下流程：

- `add_timestamps.py`：含一對多 pair 的消歧與 unmatched 的處理
- `make_temporal_split.py`
- `audit_old_leakage.py`
- `main.py`：`gold`／`pred`／`none` 三種模式、`feature_adj`、`joint`、各種 ablation，以及 random split 的 `train` scope

Leakage check 全部 PASS。真實資料尚未執行。
