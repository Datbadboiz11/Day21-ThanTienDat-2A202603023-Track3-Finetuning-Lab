# Số liệu thực nghiệm — Lab 21

Học viên: **Thân Tiến Đạt** · MSSV: **2A202603023**

Tài liệu này được sinh từ kết quả thật. Phân tích và phản tư cá nhân nằm trong REPORT.md.

Chỉ đo được 1 ca FT thua (b). Chưa đủ 2 ca thua theo rubric; ghi đúng thực tế, không gán ca hoà thành thua.

## Cấu hình đã chạy

Tier: `T4` · Model: `unsloth/Qwen3.5-4B` · Target: 50 mẫu · Regression: 15 mẫu.

GPU thực tế: Tesla T4.

## Bằng chứng mask và token

Nguồn: results/mask_proof.json, template_check.json, token_stats.json.

| chỉ số | giá trị |
|---|---|
| mask_mode | assistant-only |
| n_supervised | 39 |
| n_total | 94 |
| supervised_fraction | 0.4149 |
| answer_is_supervised | True |
| question_is_masked | True |

Đoạn được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>

```

Template:

```json
{
  "ok": true,
  "open_tag_present": true,
  "body_present": true,
  "rendered": "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n<think>\nbuoc 1: kiem tra. buoc 2: tra loi.\n</think>\n\n4<|im_end|>\n",
  "verdict": "reasoning preserved — safe to train on traces"
}
```

Thống kê token:

```json
{
  "n": 250,
  "mean": 93.1,
  "p50": 93,
  "p95": 98,
  "p99": 100,
  "max": 101,
  "suggested_max_length": 256
}
```

## So sánh ba baseline

Nguồn: results/verdict.json; baseline đóng băng tại NB2.

| run | target | regression | format | latency_ms | n |
|---|---|---|---|---|---|
| (a) base + naive prompt | 0.0 | 0.7911 | 0.0 | 3400.3 | 50 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.0 | 1063.7 | 50 |
| (c) LoRA fine-tune | 0.97 | 0.6111 | 1.0 | 1409.2 | 50 |

## Bốn cấu hình huấn luyện

Nguồn: results/runs.csv và results/autopsy.json.

| run | placement | r | trainable_params | learning_rate | max_steps | final_loss | target | format | train_seconds | peak_vram_gb |
|---|---|---|---|---|---|---|---|---|---|---|
| correct | text-linear | 16 | 32464896 | 0.0001 | 30 | 0.6268 | 0.97 | 1.0 | 415.6 | 8.78 |
| attn_only | attn-only | 283 | 32456704 | 0.0001 | 30 | 0.5366 | 0.97 | 1.0 | 276.8 | 8.79 |
| wrong_lr | text-linear | 16 | 32464896 | 1e-05 | 30 | 1.5702 | 0.0 | 0.0 | 405.1 | 8.78 |
| qlora | text-linear | 16 | 32464896 | 0.0001 | 30 | 0.7058 | 0.94 | 1.0 | 474.1 | 3.86 |

Lịch sử loss: results/training_log_<run>.json. Xếp hạng bằng target, không bằng train loss.

## Cổng hồi quy: FAILED

```json
{
  "passed": false,
  "reasons": [
    "general capability regressed by 0.180 (tolerance 0.020). See deck §6.3 — add 1-5% replay data."
  ],
  "target_delta": 0.20499999999999996,
  "regression_delta": -0.18000000000000005
}
```

## Ví dụ đối chiếu với (b)

33 ca thắng, 1 ca thua, 16 ca hoà. Nguồn: results/qualitative.json.

| i | ticket | label | baseline_b_pred | ft_pred | base_score | ft_score | outcome |
|---|---|---|---|---|---|---|---|
| 26 | Alo shop, mình đặt máy xay sinh tố mã đơn VN724342. Trả lại tiền. Không vội. Nhờ shop kiểm tra. | {"intent": "hoan_tien", "urgency": "thap", "product": "máy xay sinh tố", "sentiment": "trung_tinh"} | {"intent": "hoan_tien", "urgency": "thap", "product": "máy xay sinh tố", "sentiment": "trung_tinh"} | {"intent": "doi_tra", "urgency": "thap", "product": "máy xay sinh tố", "sentiment": "trung_tinh"} | 1.0 | 0.75 | loss |
| 36 | Chào shop, mình đặt tai nghe bluetooth mã đơn VN161530. Giá bao nhiêu. Hỏi cho biết thôi. Rất thất vọng. | {"intent": "hoi_thong_tin", "urgency": "thap", "product": "tai nghe bluetooth", "sentiment": "tieu_cuc"} | {"intent": "doi_tra", "urgency": "trung_binh", "product": "tai nghe bluetooth", "sentiment": "tieu_cuc"} | {"intent": "hoi_thong_tin", "urgency": "thap", "product": "tai nghe bluetooth", "sentiment": "tieu_cuc"} | 0.5 | 1.0 | win |
| 49 | Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm nhé. Shop xem giúp. | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"} | {"intent": "san_pham_loi", "urgency": "cao", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"} | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"} | 0.5 | 1.0 | win |
| 2 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều. | {"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"} | 1.0 | 1.0 | tie |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | {"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"} | 0.75 | 0.75 | tie |

## Đọc cùng báo cáo chính

REPORT.md chứa phần giải thích đối chứng, diễn giải phán quyết, kết luận và giới hạn thí nghiệm; REFLECTION.md ghi phản tư và việc sử dụng AI. Các bảng ở đây chỉ được sinh từ số đo, không thay thế phân tích. Không thay đổi eval hay ngưỡng chấm để tạo kết quả đẹp.
