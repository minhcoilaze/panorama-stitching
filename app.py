"""Streamlit UI for custom-core planar/cylindrical high-resolution panoramas."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import streamlit as st
from panorama import PanoramaError, PanoramaStitcher, StitchConfig
from panorama.geometry import make_thumbnail
from panorama.image_io import bgr_to_rgb, decode_image, encode_image

st.set_page_config(page_title='Panorama Stitching', page_icon='🖼️', layout='wide')
st.markdown('''<style>.block-container {padding-top:2rem;padding-bottom:3rem;}
             [data-testid="stMetricValue"] {font-size:1.35rem;}</style>''', unsafe_allow_html=True)
st.title('Ghép ảnh toàn cảnh')
st.caption('Phẳng / Hình trụ / Tự động · thuật toán lõi tự cài đặt · xuất đến 150 triệu pixel')

with st.sidebar:
    st.header('Cấu hình')
    projection_label = st.selectbox('Phép chiếu panorama', ['Tự động', 'Phẳng', 'Hình trụ'],
        key='projection', help='Tự động so sánh biến dạng, sai số và vùng crop của hai mô hình.')
    resolution_label = st.selectbox('Độ phân giải ảnh xuất',
        ['100 triệu pixel', '150 triệu pixel', '80 triệu pixel', 'Tự nhiên'], key='output_resolution',
        help='Kích thước sau crop. Nếu vượt độ phân giải tự nhiên, ảnh có nội suy; không tăng chi tiết thật.')
    auto_crop = st.toggle('Tự động cắt viền trống', value=True, key='auto_crop',
        help='Tìm hình chữ nhật kín trong mask hợp lệ; mất một phần cảnh sát mép.')
    feature_method = st.selectbox('Bộ phát hiện đặc trưng', ['SIFT', 'ORB'])
    ratio_threshold = st.slider('Lowe ratio threshold', .60, .90, .75, .01)
    ransac_threshold = st.slider('Ngưỡng sai số RANSAC (pixel)', 1., 10., 4., .5)
    max_dimension = st.select_slider('Cạnh ảnh tối đa để tìm đặc trưng',
        options=[960, 1280, 1600, 1920, 2560], value=1920,
        help='Chỉ giảm ảnh phân tích; ảnh xuất được dựng từ ảnh gốc.')
    exposure = st.toggle('Cân bằng sáng vùng chồng lấn', value=True)
    st.caption('Ảnh lớn được dựng theo vùng và lưu tạm trên đĩa. 150 triệu pixel cần ít nhất khoảng 450 MB cho ảnh chưa nén, ngoài bộ nhớ xử lý.')

uploaded = st.file_uploader('Chọn từ 2 đến 10 ảnh JPG/PNG', type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True, help='Ảnh nên chồng lấn 30–50%, chụp quanh cùng một tâm và cùng tiêu cự.')
if not uploaded:
    st.info('Chọn ảnh để bắt đầu. Có thể tạo bộ thử bằng `python scripts/generate_sample_data.py`.')
    st.stop()
if len(uploaded) > 10:
    st.error('Ứng dụng chỉ nhận tối đa 10 ảnh trong một lần ghép.')
    st.stop()

images, names, fingerprints = [], [], []
try:
    for item in uploaded:
        data = item.getvalue()
        names.append(item.name)
        fingerprints.append((item.name, hashlib.sha256(data).hexdigest()))
        images.append(decode_image(data, item.name))
except PanoramaError as error:
    st.error(str(error))
    st.stop()

st.subheader('Ảnh đầu vào')
columns = st.columns(min(len(images), 4))
for i, (image, name) in enumerate(zip(images, names)):
    with columns[i % len(columns)]:
        st.image(bgr_to_rgb(make_thumbnail(image, 500)), caption=f'{i+1}. {name}', use_container_width=True)

sorting = st.radio('Cách sắp xếp ảnh', ['Tự động', 'Thủ công'], horizontal=True)
manual_order = None
if sorting == 'Thủ công':
    st.caption('Nhập thứ tự từ trái sang phải, mỗi vị trí chỉ xuất hiện một lần.')
    columns = st.columns(min(len(images), 4))
    positions = []
    for i, name in enumerate(names):
        with columns[i % len(columns)]:
            positions.append(int(st.number_input(name, min_value=1, max_value=len(images),
                value=i+1, step=1, key=f'position-{i}-{name}')))
    if sorted(positions) == list(range(1, len(images)+1)):
        manual_order = sorted(range(len(images)), key=lambda i: positions[i])
    else:
        st.warning('Các vị trí phải không trùng và phủ đủ từ 1 đến số lượng ảnh.')

with st.expander('Tùy chọn hiển thị và tải xuống'):
    output_format = st.radio('Định dạng', ['PNG', 'JPG'], horizontal=True)
    show_intermediate = st.toggle('Hiển thị các bước trung gian', value=True)

config = StitchConfig(
    feature_method=feature_method, ratio_threshold=ratio_threshold,
    ransac_threshold=ransac_threshold, max_dimension=max_dimension,
    exposure_compensation=exposure, auto_crop=auto_crop,
    projection={'Tự động': 'auto', 'Phẳng': 'planar', 'Hình trụ': 'cylindrical'}[projection_label],
    output_megapixels={'100 triệu pixel': 100., '150 triệu pixel': 150.,
                      '80 triệu pixel': 80., 'Tự nhiên': None}[resolution_label])
signature = hashlib.sha256(json.dumps([fingerprints, asdict(config), sorting, manual_order],
                                     sort_keys=True).encode()).hexdigest()
can_process = len(images) >= 2 and (sorting == 'Tự động' or manual_order is not None)
if st.button('Bắt đầu ghép ảnh', type='primary', disabled=not can_process):
    # Drop the previous large result and encoded bytes before allocating another.
    for key in ('panorama_result', 'panorama_signature', 'panorama_download'):
        st.session_state.pop(key, None)
    bar = st.progress(0, text='Đang chuẩn bị...')
    try:
        result = PanoramaStitcher(config).stitch(images, names,
            automatic_order=sorting == 'Tự động', manual_order=manual_order,
            progress=lambda value, message: bar.progress(round(value*100), text=message))
    except PanoramaError as error:
        st.error(f'Không thể ghép ảnh: {error}')
        st.info('Xem nguyên nhân cụ thể ở trên. Với cảnh quét ngang rộng, thử Hình trụ hoặc Tự động; với thiếu overlap, kiểm tra ảnh và thứ tự.')
    except (MemoryError, OSError):
        st.error('Không đủ RAM hoặc dung lượng đĩa tạm cho kích thước đã chọn. Ứng dụng không tự hạ độ phân giải.')
    except Exception as error:
        st.error(f'Lỗi ngoài dự kiến: {error}')
    else:
        st.session_state.panorama_result = result
        st.session_state.panorama_signature = signature
    finally:
        bar.empty()

result = st.session_state.get('panorama_result')
if result is None:
    st.stop()
if st.session_state.get('panorama_signature') != signature:
    st.info('Ảnh hoặc cấu hình đã thay đổi. Bấm Bắt đầu ghép ảnh để cập nhật kết quả.')
    st.stop()

st.success('Ghép ảnh thành công.')
st.info(result.projection_reason)
for warning in result.warnings:
    st.warning(warning)
st.image(bgr_to_rgb(make_thumbnail(result.panorama, 1600)), caption='Ảnh xem trước; file tải xuống giữ độ phân giải đã chọn.',
         use_container_width=True)
columns = st.columns(5)
columns[0].metric('Số ảnh', len(result.ordered_names))
columns[1].metric('Thời gian ghép', f'{result.elapsed_seconds:.2f} s')
columns[2].metric('Kích thước', f'{result.panorama.shape[1]} × {result.panorama.shape[0]}')
columns[3].metric('Triệu pixel', f'{result.panorama.shape[0]*result.panorama.shape[1]/1e6:.2f}')
columns[4].metric('Phép chiếu', 'Hình trụ' if result.projection == 'cylindrical' else 'Phẳng')
st.write('**Thứ tự đã dùng:** '+' → '.join(result.ordered_names))
if result.coverage_verified:
    st.caption(f'Đã kiểm tra vùng crop kín ở độ phân giải xuất. Tỷ lệ vùng hợp lệ giữ lại: {result.crop_fraction:.1%}.')
else:
    st.caption('Đang giữ toàn bộ phạm vi cảnh; mép ảnh có thể còn khoảng trống.')

extension = '.png' if output_format == 'PNG' else '.jpg'
cache = st.session_state.get('panorama_download')
if cache is None or cache[0] != extension:
    # Keep only one encoded format; displaying metadata never converts the
    # full 100 MP image to RGB or recomputes the panorama.
    st.session_state.pop('panorama_download', None)
    try:
        with st.spinner('Đang mã hóa ảnh để tải xuống...'):
            encoded = encode_image(result.panorama, extension)
        st.session_state.panorama_download = (extension, encoded)
    except (PanoramaError, MemoryError) as error:
        st.error(f'Chưa thể mã hóa ảnh: {error}')
cache = st.session_state.get('panorama_download')
if cache is not None:
    st.download_button('Tải ảnh panorama', data=cache[1],
        file_name=f'panorama_{result.projection}{extension}',
        mime='image/png' if extension == '.png' else 'image/jpeg', on_click='ignore', type='primary')

if show_intermediate:
    with st.expander('Đặc trưng được phát hiện'):
        for name, visual in zip(result.ordered_names, result.keypoint_visualizations):
            st.image(bgr_to_rgb(visual), caption=name, use_container_width=True)
    with st.expander('Matching KNN và DLT/RANSAC tự cài đặt'):
        for pair in result.pair_diagnostics:
            st.write(f'Ảnh {pair.source_position+1} → {pair.destination_position+1}: '
                     f'{pair.inlier_count}/{pair.match_count} inlier; sai số {pair.mean_error:.2f} px')
            st.image(bgr_to_rgb(pair.visualization), use_container_width=True)
    with st.expander('Ảnh sau phép chiếu'):
        for name, visual in zip(result.ordered_names, result.warped_previews):
            st.image(bgr_to_rgb(visual), caption=name, use_container_width=True)
    with st.expander('Thông số đánh giá phép chiếu'):
        st.json(result.quality)
        if result.camera_focals:
            st.write('Tiêu cự ước lượng trong hệ tọa độ ảnh phân tích:', result.camera_focals)
