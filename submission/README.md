# Bài nộp Lab 21 — Thân Tiến Đạt · 2A202603023

## Đọc bài

1. `submission/REPORT.md`: báo cáo chính, đầy đủ phương pháp, số đo, phán quyết và ví dụ.
2. `submission/REFLECTION.md`: phản tư và khai báo việc sử dụng AI assistant.
3. `submission/MEASURED_RESULTS.md`: các bảng sinh trực tiếp từ kết quả NB1–NB5.
4. `submission/figures/`: biểu đồ so sánh baseline và đường loss, được vẽ từ JSON đã lưu.
5. `submission/VERIFICATION.txt`: đầu ra kiểm tra ở lần đóng gói cuối.

**Kết quả:** correct đạt target=0.97, regression=0.6111; baseline prompt tối ưu đạt
target=0.765, regression=0.7911. Cổng hồi quy FAILED do giảm 0.18, vượt ngưỡng 0.02.
Điểm target đo theo từng trường, không phải tỷ lệ ticket đúng toàn bộ.

## Đối chiếu với rubric

| Yêu cầu | Bằng chứng / vị trí trong báo cáo |
|---|---|
| Mask đúng, template, p95 | REPORT §2; mask_proof.json, template_check.json, token_stats.json |
| Adapter correct và loss/VRAM | REPORT §4; runs.csv; adapters/correct/ trong Option A |
| Ngân sách tham số và step công bằng | REPORT §4; bốn run 30 step; sai lệch tham số 0.02523% |
| Mỗi đối chứng một biến | REPORT §4.1–4.3; vị trí, LR và precision base |
| Baseline tốt trước khi train | REPORT §3; baselines_frozen.json và baseline_predictions.json |
| Đủ bốn nhóm và diễn giải verdict | REPORT §3, §5; verdict.json |
| Ít nhất năm ví dụ, có mặt thua | REPORT §6; năm ticket target và ba câu regression |
| Kết luận và phản tư | REPORT §7–8; REFLECTION.md |
| Số liệu có thể đối chiếu | results/ và SUBMISSION_MANIFEST.json |

**Giới hạn cần giữ khi chấm:** chỉ có một ca target FT thua (b), dù có thêm ca thua
thực trên regression. Nếu rubric yêu cầu hai ca thua đều là target, phần này chưa đủ.
`max_length=1024` giữ theo tier trong khi p95=98 và hàm lab gợi ý 256; REPORT §2 giải
thích lựa chọn và giới hạn, không nhận rằng 1024 là cấu hình tối ưu.

NB6 và các phần thưởng chưa thực hiện; không có số liệu hoặc điểm thưởng cho chúng.

## Chọn file ZIP để nộp

- **Option A:** `LAB21_SUBMISSION_2A202603023_OPTION_A.zip`, có adapter correct.
- **Option C:** `LAB21_SUBMISSION_2A202603023_OPTION_C.zip`, không có trọng số, dùng khi
  giới hạn dung lượng; đây là định dạng code-only được rubric cho phép.

Nộp **một** trong hai ZIP. Gói gồm báo cáo, toàn bộ kết quả, notebook không có output,
mã, test và dữ liệu public để kiểm tra. Các adapter đối chứng vẫn giữ trong dự án gốc,
nhưng không cần trong hai định dạng nộp này. Base model không đóng gói.

## Kiểm tra lại và tái lập

Giải nén ZIP rồi chạy trong thư mục `lab21_2A202603023`:

```bash
python -m pip install -r requirements-cpu.txt
python scripts/verify.py
```

Lệnh trên kiểm tra kết quả có sẵn và unit test; không tải trọng số hoặc train lại.
Trong Option C, verifier gốc không kiểm tra trọng số adapter, nên việc không có
adapter là đặc điểm định dạng nộp, không phải chứng minh rằng adapter còn ở đó.

Để train lại, dùng Linux/Colab T4, Python tương thích và cấu hình T4/assistant-only/
2 epoch, không đặt EVAL_LIMIT. Thư viện core thực tế đã pin trong
`requirements-reproduce.txt`, toàn bộ môi trường được ghi trong
`results/requirements-lock.txt`. Wheel torch `2.11.0+cu130` thuộc môi trường CUDA 13;
nếu pip mặc định không tìm được, dùng nguồn wheel CUDA phù hợp hoặc runtime đã có
đúng torch. Không cài toàn bộ freeze của Colab lên Windows.

Dùng một bản sao dự án cho thí nghiệm mới; giữ các kết quả đóng băng của bài nộp.
Hướng dẫn upload và thứ tự notebook ở `COLAB_GUIDE.md`. Chạy lại toàn bộ thí nghiệm
có thể có khác biệt do thiết bị, phiên bản phụ thuộc và tính không xác định số học.

Biểu đồ có thể tạo lại bằng `python scripts/plot_results.py` (cần matplotlib).
Đóng gói lại bằng `python scripts/package_submission.py`; script kiểm tra trước khi
tạo ZIP, có checksum từng file và kiểm tra dữ liệu trong archive sau khi nén.
