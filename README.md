# Panorama Stitching

Ứng dụng Python 3.11 / Streamlit ghép 2–10 ảnh JPG/PNG, có ba chế độ
**Phẳng / Hình trụ / Tự động** và xuất **80 / 100 / 150 triệu pixel** hoặc độ
phân giải tự nhiên. Mặc định: Tự động, 100 MP, tự cắt hình chữ nhật kín.

## Chạy trên Windows

1. Cài Python 3.11 64-bit và Python Launcher.
2. Chạy `setup_windows.bat` để tạo/cập nhật `.venv` và thư viện.
3. Chạy `run_windows.bat`, mở `http://localhost:8501`.

Nếu đã có môi trường từ phiên bản cũ, cập nhật một lần:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Không cần thay đổi PowerShell execution policy. `threadpoolctl` giới hạn số luồng
BLAS. Numba chỉ biên dịch vòng lặp warp/nội suy/blending được viết trong
`kernels.py`; không cung cấp thuật toán ghép ảnh có sẵn. Lần đầu có thể cần thêm
thời gian biên dịch; các lần sau dùng cache. Bản NumPy giữ lại làm đối chứng.

## Pipeline và thuật toán

1. Decode ảnh, chuẩn hóa hướng EXIF và giữ bản gốc.
2. Resize riêng bản đăng ký; SIFT hoặc ORB tìm keypoint/descriptor.
3. **KNN tự viết** bằng NumPy theo block: Euclidean cho SIFT, Hamming cho ORB;
   Lowe ratio test và mutual check. Không dùng BFMatcher/knnMatch.
4. **Normalized DLT tự viết**: Hartley normalization, xây ma trận thiết kế,
   tìm null-space bằng SVD; **RANSAC tự viết** với mẫu 4 điểm, loại mẫu suy biến,
   dừng thích nghi và refit consensus.
5. Sắp thứ tự thủ công hoặc **Held–Karp dynamic programming** trên số inlier.
6. **Tinh chỉnh chung hình học tự viết** với finite-difference Jacobian,
   Huber IRLS và Levenberg–Marquardt damping; chỉ dùng NumPy giải hệ tuyến tính.
7. Phẳng: joint symmetric-transfer homographies, cố định ảnh tham chiếu.
   Hình trụ: ước lượng focal bằng ràng buộc rotation của homography, khởi tạo
   camera rotations bằng SVD, bundle adjustment focal/rotation, căn up-vector.
8. Tự động: đánh giá Jacobian tại lưới điểm để đo scale stretch/anisotropy,
   sai số camera và tỷ lệ crop, rồi giải thích lựa chọn trong giao diện.
9. Dựng bản nháp nhỏ: median gain vùng chồng lấn; **Dinic max-flow/min-cut tự
   viết** tìm đường nối; feather trọng số ownership bằng box-filter tự viết.
10. **Largest all-valid rectangle tự viết** bằng histogram + monotonic stack,
    mask được co biên trước crop để bảo vệ vùng nội suy.
11. Định kích thước xuất sau crop. **Inverse mapping + bilinear interpolation
    tự viết** lấy mẫu trực tiếp ảnh gốc theo tile, với tỷ lệ resize thực tế riêng
    từng chiều (bao gồm quy ước tâm pixel); tính lại và kiểm tra coverage
    từng pixel. Lỗ nhỏ không thấy ở bản nháp được phản hồi vào mask và crop lại.
12. Áp gain và custom seam-aware feather blending theo tile, encode PNG/JPEG.

### Công thức camera và phép chiếu

Với R_i là camera-to-world, homography i → j là:

```text
H_ij = K_j R_j^T R_i K_i^-1
world_ray = R_i K_i^-1 [u, v, 1]^T
cyl_x = f_ref atan2(world_x, world_z)
cyl_y = f_ref world_y / sqrt(world_x^2 + world_z^2)
```

Inverse cylindrical mapping lấy ray `[sin(theta), cyl_y/f_ref, cos(theta)]`,
quay ngược về camera và chiếu về ảnh gốc. Không chiếu lại panorama phẳng đã giãn.
Góc được unwrap theo chuỗi để tránh nhảy ±π; ray sau camera bị loại.

### Phần dùng thư viện

OpenCV chỉ dùng cho IO/EXIF, SIFT/ORB, resize thumbnail/ảnh đăng ký và hình
minh họa keypoint/match. NumPy cung cấp phép toán mảng và đại số tuyến tính.
Pipeline sản phẩm **không gọi** `cv2.Stitcher`, `cv2.findHomography`,
`cv2.warpPerspective`, `cv2.remap`, `cv2.Rodrigues`, bộ tối ưu đóng gói,
seam finder hoặc blender có sẵn. Có test AST kiểm tra ràng buộc này.

## Chọn chế độ

- **Tự động:** mặc định; phân tích hai mô hình ở độ phân giải nhỏ rồi chỉ dựng
  mô hình được chọn ở độ phân giải xuất. Không quyết định dựa vào dung lượng JPG.
- **Phẳng:** giữ đường thẳng khi hình học phù hợp. Góc nhìn quá rộng có thể giãn
  mạnh ở hai đầu; chương trình cảnh báo nhưng tôn trọng lựa chọn thủ công.
- **Hình trụ:** phù hợp quét ngang rộng quanh một tâm; một số đường ngang có thể
  cong. Mô hình cần focal/hướng camera đáng tin cậy. Nếu chỉ có tịnh tiến hoặc
  camera fit không ổn định, báo lỗi cụ thể, không tự giả định focal tùy ý.

Camera dùng tỷ lệ focal/kích thước ảnh chung cho chuỗi cùng máy, cùng tiêu cự;
không xem đây là mô hình cho nhiều camera/zoom tùy ý. Không cần EXIF cho focal.

## Độ phân giải, crop và bộ nhớ

- 100 MP là **mục tiêu xuất sau crop**, không phải hứa hẹn có 100 MP chi tiết gốc.
  Nếu cần phóng lớn, UI ghi **Có nội suy**. Số pixel thực tế có thể nhỉnh hơn
  mục tiêu một hàng do làm tròn kích thước nguyên.
- Bản đăng ký 960/1280/1920 không giới hạn kích thước xuất: renderer lấy ảnh gốc.
- Bật crop: tìm hình chữ nhật không có khoảng trống, chấp nhận mất cảnh ở mép.
  Tắt crop: giữ full phạm vi cảnh, có thể còn biên trống. Ảnh đen thật vẫn hợp lệ.
- Mảng float chỉ tồn tại trong tile mặc định 96×1024; output lớn hơn 24 MP được
  lưu memmap trong thư mục tạm của Windows, tự xóa khi kết quả được giải phóng.
- Chỉ ảnh 8-bit 100 MP đã cần khoảng 300 MB; 150 MP khoảng 450 MB. Mã hóa và
  trình duyệt còn cần bộ nhớ bổ sung. Nếu thiếu tài nguyên, báo lỗi và **không
  âm thầm giảm độ phân giải**. File tạm nằm trên đĩa có thể làm quá trình chậm hơn.
- Giao diện chỉ chuyển thumbnail sang RGB; giữ kết quả qua các lần rerun và cache
  một định dạng tải xuống tại một thời điểm.

## Cấu trúc

```text
app.py                    Giao diện và kết quả giữ trong session
panorama/
  config.py               Chế độ, target MP, crop, kích thước tile
  features.py             SIFT/ORB và KNN/ratio/mutual tự viết
  homography.py           DLT/RANSAC tự viết
  ordering.py             Held–Karp ordering và cache pair
  optimization.py         Robust LM/Huber tự viết
  camera.py               Focal/rotation calibration, joint geometry refinement
  projection.py           Planar/cylindrical và lựa chọn tự động
  cropping.py             Largest rectangle trong mask
  seams.py                Dinic min-cut, seam ownership và feather weights
  rendering.py            Draft, gain, crop và tiled bilinear renderer
  kernels.py              Vòng lặp renderer tự viết, biên dịch bằng Numba
  stitcher.py             Điều phối và metadata kết quả
  geometry.py, image_io.py, exceptions.py
scripts/
  generate_sample_data.py Bộ ảnh tổng hợp
  diagnose_dataset.py     Chạy bộ thật, xuất ảnh/preview/JSON đo thời gian và RAM
tests/                    Unit, integration và Streamlit AppTest
docs/superpowers/         Thiết kế, kế hoạch và nhật ký triển khai
```

## Kiểm thử và chẩn đoán

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\generate_sample_data.py
.\.venv\Scripts\python.exe scripts\diagnose_dataset.py 'C:\Code\ngoai_nha\1' --mode auto --mp 100
.\.venv\Scripts\python.exe scripts\diagnose_dataset.py 'C:\Code\ngoai_nha\1' --mode cylindrical --mp 150
```

Chẩn đoán ghi output và report vào `outputs/`, không thay đổi ảnh đầu vào.
Thêm `--mp native`, `--mode planar`, `--manual`, `--no-crop`, `--format png`
để đối chiếu các luồng. Tests nhỏ dùng MP thấp để kiểm tra nhanh; khả năng xuất
100/150 MP phải được xác nhận bằng chạy thật và báo cáo riêng.

## Giới hạn thực tế

Chụp quanh cùng một tâm, giữ tiêu cự, overlap khoảng 30–50%, tránh rung/mờ.
Cây lá, người/xe chuyển động, vật gần và camera tịnh tiến có thể còn parallax
hoặc bóng ma. Seam finding/feather không thể bảo đảm sửa mọi sai lệch 3-D.
Độ giãn thấp và all-valid crop không thay thế việc xem chất lượng trực quan.

## Tài liệu nền

- Brown & Lowe, Automatic Panoramic Image Stitching using Invariant Features:
  https://mattabrown.github.io/pdf/ijcv2007.pdf
- OpenCV detailed stitching sample (tham khảo kiến trúc, không gọi pipeline):
  https://docs.opencv.org/4.13.0/d9/dd8/samples_2cpp_2stitching_detailed_8cpp-example.html
- Szeliski, Image Alignment and Stitching: A Tutorial:
  https://www.microsoft.com/en-us/research/publication/image-alignment-and-stitching-a-tutorial/
