import streamlit as st
import os
from modules.models import *
from modules.ui import *

# 설정 및 초기화
st.set_page_config(page_title="MNIST Digit Recognizer", layout="wide")
apply_css()

model = load_and_check()

SAVE_DIR = "saved_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# 2x2 레이아웃 생성
row1_cols, row2_cols = layout_setup()
input_col, processed_col = row1_cols
result_col, output_col = row2_cols

# 캔버스 입력
canvas_result = display_canvas(input_col)

# 전처리 이미지 출력
preprocessed = None
if canvas_result and canvas_result.image_data is not None:
    preprocessed = display_processed_image(processed_col, canvas_result)

# 결과 확률 분포 출력
if preprocessed is not None:
    display_results(result_col, preprocessed)

# 최종 예측 및 이미지 저장
if preprocessed is not None:
    display_output(output_col, preprocessed, SAVE_DIR)

st.markdown("---")

# 최근 저장 이미지 보기
display_saved_images(st, SAVE_DIR)
