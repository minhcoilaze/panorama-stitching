from pathlib import Path
from streamlit.testing.v1 import AppTest
import streamlit as st
from panorama.image_io import encode_image
from scripts.generate_sample_data import build_scene, create_crops


def test_app_has_three_modes_and_high_resolution_controls():
    app = AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run(timeout=20)
    assert not app.exception
    assert app.selectbox(key='projection').options == ['Tự động', 'Phẳng', 'Hình trụ']
    assert app.selectbox(key='output_resolution').value == '100 triệu pixel'
    assert app.toggle(key='auto_crop').value


def test_real_upload_flow_preserves_result_when_changing_download_format(monkeypatch):
    class Uploaded:
        def __init__(self, name, data):
            self.name, self.data = name, data
        def getvalue(self):
            return self.data
    images = create_crops(build_scene(width=1500, height=540, seed=22), count=2)
    uploaded = [Uploaded(f'{i}.png', encode_image(im)) for i, im in enumerate(images)]
    monkeypatch.setattr(st, 'file_uploader', lambda *a, **k: uploaded)
    app = AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run(timeout=30)
    app.selectbox(key='output_resolution').select('Tự nhiên')
    app.selectbox[2].select('ORB')
    app.run(timeout=30)
    app.button[0].click().run(timeout=40)
    assert not app.exception
    assert not app.error
    result = app.session_state['panorama_result']
    assert result.coverage_verified
    signature = app.session_state['panorama_signature']
    app.radio[1].set_value('JPG').run(timeout=20)
    assert not app.exception
    assert app.session_state['panorama_result'] is result
    assert app.session_state['panorama_signature'] == signature
    assert app.session_state['panorama_download'][0] == '.jpg'
