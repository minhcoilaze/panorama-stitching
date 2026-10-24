# HỆ THỐNG GHÉP ẢNH TOÀN CẢNH (PANORAMA STITCHING)
**Môn học:** Xử lý ảnh số

---

## 1. THÔNG TIN CHUNG

| Mục | Nội dung |
|---|---|
| Tên đề tài | Xây dựng hệ thống ghép ảnh toàn cảnh (Panorama Stitching) |
| Loại đề tài | Bài tập lớn môn Xử lý ảnh số |
| Phạm vi kỹ thuật | Kết hợp tự cài đặt thuật toán lõi và sử dụng thư viện hỗ trợ (OpenCV) |
| Sản phẩm bàn giao | Báo cáo, Slide thuyết trình, Source code, Ứng dụng hoàn chỉnh (có giao diện) |

---

## 2. MỤC TIÊU DỰ ÁN

### 2.1. Mục tiêu tổng quát
Xây dựng một ứng dụng có khả năng ghép nhiều ảnh chồng lấn (chụp từ các góc khác nhau quanh một tâm) thành một ảnh toàn cảnh (panorama) liền mạch, tự nhiên, có giao diện sử dụng thân thiện.

### 2.2. Mục tiêu cụ thể
1. **Về học thuật:** Hiểu và triển khai được quy trình xử lý ảnh cốt lõi: phát hiện đặc trưng, khớp đặc trưng, ước lượng homography, biến đổi hình học và trộn ảnh — tối thiểu tự cài đặt được 2/5 bước bằng code thuần (không gọi thẳng hàm thư viện), phần còn lại được phép dùng OpenCV.
2. **Về sản phẩm:** Hoàn thành ứng dụng chạy được, nhận đầu vào là 2–10 ảnh và xuất ra 1 ảnh toàn cảnh, xử lý thành công với thời gian chấp nhận được (< 30 giây cho bộ 5 ảnh độ phân giải trung bình).
3. **Về chất lượng:** Ảnh kết quả không bị lệch sáng rõ rệt, không có "bóng ma" (ghosting) ở vùng chồng lấn, tỷ lệ ghép thành công ≥ 80% trên bộ dữ liệu thử nghiệm tự thu thập.
4. **Về tiến độ:** Hoàn thành đúng hạn trong 5–8 tuần theo các mốc (milestone) đã đề ra ở mục 5.

---

## 3. NHIỆM VỤ

| # | Nhiệm vụ | Mô tả | Công cụ/Kỹ thuật |
|---|---|---|---|
| 1 | Nghiên cứu lý thuyết | Tìm hiểu lý thuyết feature detection, homography, RANSAC, blending | Tài liệu, paper, OpenCV docs |
| 2 | Thu thập dữ liệu | Chụp/tổng hợp bộ ảnh test (nhiều bối cảnh: trong nhà, ngoài trời, có/không vật chuyển động) | Điện thoại/máy ảnh |
| 3 | Phát hiện & khớp đặc trưng | Cài đặt hoặc tích hợp SIFT/ORB, feature matching + Lowe's ratio test | Python, OpenCV |
| 4 | Ước lượng Homography | Cài đặt RANSAC (tự viết hoặc dùng `cv2.findHomography`) | Python, NumPy |
| 5 | Biến đổi & căn chỉnh ảnh | Warping ảnh về hệ tọa độ chung | `cv2.warpPerspective` |
| 6 | Trộn ảnh (Blending) | Xử lý vùng chồng lấn: feathering / multi-band blending / seam finding | Tự cài đặt (điểm nhấn kỹ thuật) |
| 7 | Hậu xử lý | Cắt biên, cân bằng màu/sáng | OpenCV |
| 8 | Xây dựng giao diện | Giao diện cho phép người dùng chọn ảnh, xem tiến trình, tải kết quả | Streamlit/PyQt/Web (tùy chọn) |
| 9 | Kiểm thử & đánh giá | Chạy trên nhiều bộ dữ liệu, đo thời gian xử lý, đánh giá chất lượng | Thủ công + thống kê |
| 10 | Viết báo cáo & slide | Trình bày lý thuyết, kết quả, khó khăn, hướng phát triển | Word/LaTeX, PowerPoint |

---

## 4. YÊU CẦU DỰ ÁN

### 4.1. Yêu cầu chức năng
- Cho phép người dùng tải lên từ 2 ảnh trở lên.
- Tự động sắp xếp thứ tự ảnh (hoặc yêu cầu người dùng sắp theo thứ tự chụp).
- Xuất ra ảnh toàn cảnh dưới định dạng JPG/PNG.
- Hiển thị được các bước trung gian (tùy chọn, cộng điểm kỹ thuật): keypoints, matches, ảnh sau warp.
- Cho phép tải xuống kết quả.

### 4.2. Yêu cầu phi chức năng
- **Hiệu năng:** xử lý bộ 5 ảnh (độ phân giải ~1920×1080) trong dưới 30 giây trên máy cá nhân thông thường.
- **Độ ổn định:** xử lý được cả trường hợp lỗi (ảnh không đủ điểm khớp, ảnh không chồng lấn) mà không crash chương trình — cần có thông báo lỗi rõ ràng.
- **Khả năng mở rộng:** cấu trúc code tách module rõ ràng (feature detection, matching, homography, blending) để dễ thay thế/nâng cấp.
- **Giao diện:** trực quan, tối thiểu là giao diện desktop/web đơn giản, không bắt buộc đẹp nhưng phải dùng được.

### 4.3. Yêu cầu đầu vào/đầu ra
- **Đầu vào:** tập ảnh (2–10 ảnh) chụp từ cùng một vị trí, chồng lấn nhau 20–50%.
- **Đầu ra:** 1 ảnh toàn cảnh + (tùy chọn) ảnh minh họa các bước trung gian phục vụ báo cáo.

### 4.4. Ràng buộc
- Ngôn ngữ lập trình: Python (khuyến nghị, do hệ sinh thái OpenCV mạnh).
- Phải có phần **tự cài đặt tối thiểu 1 thuật toán lõi** (không chỉ gọi hàm `cv2.Stitcher`) để thể hiện hiểu bản chất môn học.
- Không dùng ảnh có bản quyền không rõ nguồn gốc trong bộ test nộp kèm báo cáo.

---

## 5. TIẾN ĐỘ DỰ KIẾN

| Tuần | Mốc | Kết quả cần đạt |
|---|---|---|
| 1 | Khởi động & nghiên cứu | Xác định phạm vi, phân công vai trò, thu thập tài liệu tham khảo |
| 2 | Thu thập dữ liệu + Feature detection | Bộ ảnh test sẵn sàng, code phát hiện & khớp đặc trưng chạy được |
| 3–4 | Homography + Warping | Ghép được 2 ảnh cơ bản, demo kết quả trung gian |
| 5 | Blending + hậu xử lý | Ghép được nhiều ảnh (>2), giảm hiện tượng đường nối/ghosting |
| 6 | Giao diện + kiểm thử | Ứng dụng hoàn chỉnh, chạy thử trên nhiều bộ dữ liệu |
| 7–8 | Hoàn thiện & báo cáo | Viết báo cáo, làm slide, tổng duyệt, chuẩn bị thuyết trình |


---

## 6. PHÂN CÔNG VAI TRÒ

| Vai trò | Trách nhiệm chính |
|---|---|
| Thành viên 1 – Core Algorithm | Feature detection, matching, homography (RANSAC) |
| Thành viên 2 – Image Processing | Warping, blending, hậu xử lý, tối ưu chất lượng ảnh |
| Thành viên 3 – Ứng dụng & Báo cáo | Giao diện, kiểm thử, tổng hợp báo cáo/slide |


---

## 7. RỦI RO DỰ KIẾN & PHƯƠNG ÁN XỬ LÝ

| Rủi ro | Ảnh hưởng | Giải pháp |
|---|---|---|
| Ảnh không đủ điểm đặc trưng khớp | Ghép thất bại | Yêu cầu ảnh chồng lấn ≥30%, có kết cấu rõ |
| Vật thể chuyển động giữa các ảnh | Ghosting | Ưu tiên bộ ảnh tĩnh khi test, ghi nhận là hạn chế đã biết |
| Chênh lệch sáng giữa ảnh | Đường nối lộ rõ | Áp dụng multi-band blending/cân bằng exposure |
| Tiến độ trễ do phần tự cài đặt thuật toán khó | Không hoàn thành đúng hạn | Làm bản MVP bằng OpenCV trước, tự cài đặt song song, có phương án lùi (fallback) |

---

## 8. TIÊU CHÍ ĐÁNH GIÁ THÀNH CÔNG
- Ứng dụng chạy được, cho ra ảnh toàn cảnh hợp lý trên tối thiểu 3 bộ dữ liệu khác nhau.
- Có ít nhất 1 module tự cài đặt (không dùng thẳng hàm OpenCV) được giải thích rõ trong báo cáo.
- Báo cáo trình bày đầy đủ: lý thuyết, thiết kế hệ thống, kết quả thực nghiệm, hạn chế, hướng phát triển.
- Demo trực tiếp chạy ổn định khi thuyết trình.
