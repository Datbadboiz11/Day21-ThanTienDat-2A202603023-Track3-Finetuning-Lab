# Chạy Lab 21 trên Colab — Thân Tiến Đạt · 2A202603023

## Hai file cần dùng

1. `colab/Lab21_RUN_ALL.ipynb`: mở bằng Google Colab.
2. `dist/COLAB_INPUT_2A202603023.zip`: upload khi ô Setup yêu cầu.

ZIP chứa mã, notebook, test, hướng dẫn và dữ liệu. Không cần push GitHub hoặc upload
từng JSONL. Colab tự tải tokenizer/trọng số model từ Hugging Face; cần Internet.
Máy cá nhân có GTX 1650 4 GB VRAM; với cấu hình đang chọn, chạy phần GPU trên Colab T4.

## Thao tác từng bước

1. Vào Google Colab → **File → Upload notebook**, mở `Lab21_RUN_ALL.ipynb`.
2. Chọn **Runtime → Change runtime type → T4 GPU**.
3. Chạy **ô 1 — Setup**. Chọn `COLAB_INPUT_2A202603023.zip` khi hộp upload hiện ra.
   Giữ `SOURCE="upload"`. `SAVE_TO_DRIVE=True` yêu cầu kết nối Drive và sao lưu tiến độ
   vào `MyDrive/Lab21_2A202603023/`. Đặt `False` nếu không dùng Drive; cần tải ZIP về
   trước khi runtime mất dữ liệu.
4. Chạy **ô 2 — Smoke**: unit test phải pass. GPU và phiên bản thư viện được ghi tại
   `results/environment.json` và `results/requirements-lock.txt`.
5. Chạy **ô 3 — NB1 + NB2**: kiểm tra mask, đọc điểm (a)/(b). Nếu (b) chưa thắng (a),
   dừng để cải thiện prompt và đo lại NB2 trước khi train. Không làm yếu prompt (b).
6. Chạy **ô 4 — NB3 + NB4**: adapter chính và ba đối chứng; đây là ô chạy lâu nhất.
7. Chạy **ô 5 — NB5**: đánh giá và sinh bảng số liệu `submission/MEASURED_RESULTS.md`.
8. Chạy **ô 6 — Verify + Download**: tải `LAB21_RESULTS_2A202603023.zip` về máy.
   Verify còn FAIL nếu báo cáo chưa viết xong; ô vẫn tải ZIP để hoàn thiện báo cáo.
9. Gửi ZIP kết quả để hoàn thiện phần phân tích từ số đo thật. Sau đó chạy lại
   `python scripts/verify.py` và đóng gói theo rubric.

Không cần chạy thêm sáu notebook riêng sau RUN_ALL. NB6 tuỳ chọn, không bắt buộc.
Core NB1–NB5 khoảng 100–130 phút theo số đo T4 trong repo, chưa tính cài đặt/tải model.

## Data cần cho từng notebook

| Dữ liệu | Số mẫu | Notebook dùng | Vai trò |
|---|---:|---|---|
| `data/train_seed.jsonl` | 250 | NB1 | Kiểm tra mask/token, chia train/val |
| `data/eval_target.jsonl` | 50 | NB2, NB5; NB6 nếu làm | Chấm ticket → JSON bốn trường |
| `data/eval_regression.jsonl` | 15 | NB2, NB5 | Kiểm tra năng lực chung có tụt không |
| `data/checksums.json` | — | Verify | Đối chiếu tính toàn vẹn dữ liệu |
| `data/split/train.jsonl` | 225 | NB3, NB4 | NB1 tự sinh; dữ liệu đưa vào trainer |
| `data/split/val.jsonl` | 25 | NB1 sinh | Tập dự phòng; trainer hiện không chấm trên tập này |

**Bốn file gốc đã nằm trong ZIP; hai file split tự sinh sau NB1.**
`data/holdout_secret.jsonl` không cần cho NB1–NB6, không nằm trong gói upload và không
đưa vào train. Không cần thu thập dữ liệu mới cho lượt này.

Train có `instruction`, `input`, `output`. Target eval có `input`, `label`.
Regression eval có `instruction`, `keywords`. Giữ nguyên corpus mặc định.

## Nếu chạy notebook riêng

| File Colab | GPU | Cần có trước |
|---|---|---|
| `Lab21_01_data_and_mask.ipynb` | CPU đủ; Setup chọn T4 | Corpus train và tokenizer |
| `Lab21_02_baselines.ipynb` | Có | Hai tập eval, cùng model/tier với thí nghiệm |
| `Lab21_03_train_correct.ipynb` | Có | Split và mask NB1, baseline NB2 |
| `Lab21_04_misconfig_autopsy.ipynb` | Có | Split NB1, run correct NB3 để đối chiếu |
| `Lab21_05_evaluate_and_verdict.ipynb` | Có | Hai tập eval, baseline đóng băng và dự đoán, cả bốn adapter |
| `Lab21_06_merge_and_serve.ipynb` | Có; tuỳ chọn | Target eval, correct và adapter thứ hai để hot-swap |

Dùng cùng runtime hoặc upload ZIP tiến độ vào runtime mới. Mở notebook mới không tự
chuyển dữ liệu/adapter từ runtime cũ. NB6 không cần để hoàn thành core.

## Cấu hình đã chuẩn bị

- `T4`, model `unsloth/Qwen3.5-4B`, `MASK_MODE=assistant-only`, `EPOCHS=2`, seed=42.
- Toàn bộ eval: không đặt `EVAL_LIMIT`; không dùng chế độ 8 mẫu khi nộp.
- Correct: text-linear, r=16, alpha=32, LR=1e-4.
- Attn-only: q,v; rank được tính để khớp ngân sách tham số.
- Wrong-LR: giữ cấu hình chính, đổi LR=1e-5. QLoRA: đổi base sang 4-bit.
- Bốn run có cùng ngân sách step. `max_length=1024` theo tier; báo cáo phải đối chiếu
  với p95 NB1 và giải thích nếu khác độ dài đề xuất.

## Tiếp tục sau khi Colab bị ngắt

Nếu bật Drive, tải `LAB21_PROGRESS_2A202603023.zip` từ thư mục Drive nói trên.
Mở runtime T4 mới, upload ZIP tiến độ ở ô 1, chạy ô 2 rồi tiếp tục ô 3–5.
Ô 3 giữ baseline khi đã có correct; ô 4 bỏ qua correct và các đối chứng đã lưu.
NB4 sao lưu sau từng adapter. Backup giữ adapter đã train xong, không giữ optimizer
để tiếp tục giữa một run đang dở; run dở phải chạy lại.

Giữ nguyên model, dữ liệu và cấu hình khi tiếp tục. Nếu một ô lỗi, vẫn chạy ô 6 để
tải kết quả hiện có và gửi traceback để xử lý.

## Kết quả cần lấy về

ZIP gồm toàn bộ `results/`, các adapter đã hoàn thành, `data/split/`, báo cáo, mã và
notebook. Lịch sử loss lưu tại `results/training_log_<run>.json`. Output đầy đủ của
(b)/(c) nằm trong `qualitative.json`; baseline còn có `baseline_predictions.json`.

ZIP kết quả là bản sao thí nghiệm, chưa tự động là bài nộp cuối. Cần viết ba phân tích
đối chứng (mỗi câu hỏi ≥3 câu), phán quyết ≥100 từ, kết luận ≥150 từ và phản tư cá nhân.
Cần ≥5 ví dụ, có ≥2 ca FT thua (b). Nếu số đo không có đủ ca thua, ghi đúng thực tế và
nêu giới hạn; không biến ca hoà thành thua hoặc dùng holdout để tìm thêm ca thua.

## Tạo lại ZIP sau khi sửa mã trên máy cá nhân

Chạy trong thư mục gốc bằng PowerShell:

```powershell
python scripts/build_colab.py
python scripts/export_colab.py --mode input
```

Gói mới nằm tại `dist/COLAB_INPUT_2A202603023.zip`. Khi mã đổi, dùng ZIP mới và runtime
mới cho thí nghiệm mới, tránh dùng mã đã import của phiên cũ.
