# Reflection — Lab 21

**Học viên:** Thân Tiến Đạt · **MSSV:** 2A202603023

## 1. Điều gì làm tôi ngạc nhiên nhất?

Một bản fine-tune đạt 97% độ chính xác từng trường và có định dạng tốt vẫn nhận FAILED. Lý do nằm ở điểm regression giảm 18 điểm phần trăm. Câu hỏi đổi đơn vị lại được trả thành JSON triage là ví dụ khiến tôi thấy rõ việc tối ưu tác vụ hẹp có thể làm lệch hành vi của model. Tôi cũng chú ý rằng attn_only hoà correct ở target dù có loss trung bình thấp hơn.

## 2. Tôi mất nhiều thời gian ở đâu?

Phần chạy ba đối chứng NB4 có tổng thời gian train được ghi là 1156.0 giây, khoảng 19.27 phút; NB3 là 415.6 giây. Đây là phần chờ huấn luyện đáng kể được chứng minh bằng log, chưa gồm thời gian tải model và sinh eval. Tôi đã dự kiến train cần GPU và thời gian, nhưng việc đọc output regression và phân biệt ca FT sai với ca FT thua mới là phần cần suy nghĩ kỹ để kết luận đúng. Không có nhật ký đủ chi tiết để báo một con số về thời gian thao tác cá nhân hoặc debug.

## 3. Bài lab thay đổi cách tôi đánh giá fine-tuning thế nào?

Tôi không còn coi loss thấp hoặc target cao là đủ để quyết định dùng adapter. Cần một baseline prompt tốt, một tập eval giữ cố định và phép kiểm tra khả năng ngoài miền. Tôi cũng không xem rank lớn là một bảo đảm tốt hơn: attn_only phải tăng rank lên 283 để khớp tham số với correct r=16, nhưng target vẫn hoà. Mỗi kết luận phải gắn với biến đã kiểm soát và thang đo đã dùng.

## 4. Tôi dùng AI assistant vào việc gì?

AI đọc cấu trúc và rubric, chuẩn bị notebook upload ZIP, hướng dẫn dữ liệu và thao tác Colab, thêm lưu dự đoán/lịch sử loss, giải nén và đối chiếu kết quả, hỗ trợ viết báo cáo. Tôi thực hiện phần chạy Colab. Một giới hạn đã được phát hiện là chọn các ca FT điểm thấp chưa đủ chứng minh FT thua baseline; mã đã được bổ sung đối chiếu trực tiếp (b)/(c). Phần viết báo cáo phải giữ đúng số đo: chỉ một ca target thua, không biến ca hoà thành thua để khớp mẫu.

## 5. Bước đầu khi làm cho khách hàng thật?

Tôi sẽ làm rõ đầu ra cần đạt, dữ liệu đầu vào thực tế và yêu cầu giữ năng lực chung; sau đó xây baseline prompt tốt trên tập đánh giá độc lập trước khi fine-tune. Chỉ huấn luyện khi đã có cách chứng minh lợi ích và nhận ra hồi quy. Nếu tác vụ chỉ là triage, có thể giới hạn phạm vi bằng routing; nếu model phải đa dụng, phải giữ cổng regression và thử replay trong một thí nghiệm mới. Không triển khai chỉ vì nhìn thấy target 97% trên corpus nhỏ.
