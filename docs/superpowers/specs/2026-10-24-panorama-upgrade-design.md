# Thiết kế được duyệt: panorama phẳng, hình trụ và tự động

Người dùng duyệt pipeline trong hội thoại và yêu cầu các thuật toán lõi phải tự viết.

## Phạm vi
- Ba chế độ planar/cylindrical/auto; auto mặc định và giải thích lựa chọn.
- Xuất 100 MP mặc định, chọn 80/100/150 MP hoặc độ phân giải tự nhiên.
- Độ phân giải đăng ký tách khỏi dựng ảnh gốc; không âm thầm hạ độ phân giải xuất.
- Crop chữ nhật kín dựa trên mask, mặc định bật; tắt để giữ đầy đủ phạm vi cảnh.
- DLT/RANSAC, KNN matching, tối ưu robust, camera autocalibration, phép chiếu,
  inverse mapping/bilinear interpolation, chọn phép chiếu, graph-cut seam,
  feather blending, gain và largest-rectangle crop được viết trong dự án.
- Thư viện chỉ hỗ trợ đọc/ghi ảnh, SIFT/ORB, hiển thị và đại số tuyến tính cơ bản.
- Không gọi Stitcher/findHomography/warpPerspective/remap, bộ tối ưu, seam/blender
  hoặc thuật toán crop đóng gói trong pipeline sản phẩm.
- Xử lý output theo tile và lưu memmap khi lớn; xem trước bằng thumbnail.
- Sai hình học phải bị từ chối; giới hạn RAM không được tráo thành giảm chất lượng.

## Các hướng đã xét
Giảm ảnh xuống 1280 và tăng giới hạn canvas đơn thuần không xử lý kéo giãn.
Hướng được chọn: camera rotation + cylindrical projection, đồng thời giữ planar,
tự chọn bằng sai số, độ giãn và vùng nội dung giữ lại; tối ưu bộ nhớ và crop kín.

## Pipeline
Decode → ảnh đăng ký → SIFT/ORB → KNN tự viết + ratio/mutual → DLT/RANSAC
→ Held–Karp ordering → tinh chỉnh homography / camera rotation và focal bằng LM
robust → đánh giá hai phép chiếu → mask/crop → chọn mật độ đầu ra → tiled warp
→ median gain → custom graph-cut seam và custom feather → PNG/JPEG.

## Ràng buộc và nghiệm thu
Ảnh đen thật không bị crop nhầm. Camera chỉ quay là mô hình cho cylindrical;
parallax và vật chuyển động có thể vẫn cần chụp lại. Nội suy 100 MP không tạo
chi tiết mới và phải được ghi rõ trong UI. Graph-cut chạy trên bản nháp nhỏ.
Kiểm thử synthetic và thư mục C:\Code\ngoai_nha\1; auto phải chọn planar cho
ảnh crop dịch chuyển và cylindrical cho bộ ngoại cảnh nếu camera fit đủ tốt.
Kiểm tra 100/150 MP thực tế, all-valid crop, encoding và giải phóng temp file.

## Quyết định triển khai
Làm trực tiếp trong thư mục hiện tại vì không có Git; backup source dạng ZIP.
Không thêm SciPy: tự viết damped least squares với finite differences và Huber.
Mô hình camera dùng focal chuẩn hóa chung cho chuỗi cùng camera/tiêu cự; báo lỗi
nếu mô hình không phù hợp, không tuyên bố hỗ trợ chuyển động tịnh tiến tùy ý.
