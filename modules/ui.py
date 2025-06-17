import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import os
from modules.models import *

# css로 레이아웃 정의
def apply_css():
    st.markdown(
        """
        <style>
        canvas {
            background-color: white !important;
        }
        .stCanvas {
            background-color: white !important;
            padding: 0 !important;
            margin: 0 auto !important;
        }
        div[data-testid="stVerticalBlock"] {
            gap: 0rem !important;
        }
        </style>
        """, unsafe_allow_html=True
    )

# 기본 레이아웃 설정
def layout_setup():
    st.title("MNIST 숫자 분류 App")
    st.markdown("---")
    row1_cols = st.columns([1, 2])  # 입력 캔버스  / 원본 이미지 & 전처리된 이미지 비율 설정 (1, 2인 이유는 원본 이미지 & 전처리된 이미지가 2에 해당)
    row2_cols = st.columns([1, 1])  # 결과 표 / 예측 결과 비율 설정
    return row1_cols, row2_cols

# 입력 캔버스 UI 설정
def display_canvas(col):
    with col:
        st.subheader("입력 캔버스")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)", 
            stroke_width=15,
            stroke_color="black",
            background_color="white",            
            height=400,                          
            width=400,                           
            drawing_mode="freedraw",
            key="canvas",
            update_streamlit=True
        )
        st.caption("숫자를 그려주세요.")
    return canvas_result

# 입력에서 그린 이미지 & 전처리 이미지 UI 설정
def display_processed_image(col, canvas_result):
    with col:
        st.subheader("전처리 이미지")
        if canvas_result.image_data is not None:
            # 입력 이미지 (캔버스 원본)
            img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))

            # 전처리 이미지
            preprocessed = preprocess(img)
            pre_img = preprocessed.reshape(28, 28)

            # 가로로 나란히 이미지 출력
            img_col1, img_col2 = st.columns(2)

            with img_col1:
                st.image(img, caption="원본 캔버스 이미지", width=400)

            with img_col2:
                st.image(pre_img, caption="전처리 결과", clamp=True, width=400)

            return preprocessed
        else:
            st.info("캔버스에 그림을 그려주세요.")
            return None

# 결과 표 레이아웃 설정
def display_results(col, preprocessed):
    with col:
        st.subheader("결과 확률 분포")
        if preprocessed is not None:
            logits = predict(preprocessed)
            prob = softmax(logits)  # 확률 값으로 보여주기 위해 softmax
            sorted_probs = sorted(enumerate(prob), key=lambda x: x[1], reverse=True)  # 결과 정렬
            for i, (digit, prob_val) in enumerate(sorted_probs):
                prob_val = float(np.clip(prob_val, 0.0, 1.0))
                col1, col2 = st.columns([1, 5])
                with col1:
                    size = 30 if i == 0 else 22 if i == 1 else 20  # 정렬된 기준으로 1, 2번은 사이즈 조금 키움
                    weight = 700 if i == 0 else 600 if i == 1 else 500  # 동일한 이유
                    st.markdown(f"<span style='font-size: {size}px; font-weight: {weight};'>{digit}</span>", unsafe_allow_html=True)
                with col2:
                    st.progress(prob_val)
                    st.markdown(f"<div style='text-align: right; font-weight: 600;'>{prob_val * 100:.2f}%</div>", unsafe_allow_html=True)
        else:
            st.info("전처리 이미지를 먼저 생성하세요.")  # 입력 캔버스에 아무것도 안 그리면 이걸 출력

# 예측 결과 레이아웃 설정
def display_output(col, preprocessed, save_dir):
    from datetime import datetime

    with col:
        st.subheader("예측 결과")
        if preprocessed is not None:
            logits = predict(preprocessed)
            prob = softmax(logits)
            pred_label = int(np.argmax(prob))
            confidence = prob[pred_label]

            st.markdown(
                f"""
                <div class="result-box">
                    <h2>예측 결과</h2>
                    <h1>{pred_label}</h1>
                    <p>신뢰도: <strong>{confidence * 100:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True
            )  # 숫자 크게 보여주고, 예측 confidence 찍어주기

            if st.button("이미지 저장", use_container_width=True):
                img_array = (preprocessed.reshape(28, 28) * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = os.path.join(save_dir, f"{timestamp}_label{pred_label}.png")
                img.save(fname)
                st.success(f"✅ 이미지가 저장되었습니다: `{fname}`", icon="🎉")
        else:
            st.info("전처리 이미지를 먼저 생성하세요.")

# 저장된 이미지 보여주기
def display_saved_images(st, save_dir):
    with st.expander("📁 최근 저장 이미지 보기", expanded=False):
        files = sorted([f for f in os.listdir(save_dir) if f.endswith(".png")])[-10:]
        if files:
            cols = st.columns(5)
            for idx, fname in enumerate(files[::-1]):
                label = fname.split("label")[-1].split(".")[0]
                with cols[idx % 5]:
                    st.image(os.path.join(save_dir, fname), width=120, caption=f"Label: {label}")
        else:
            st.info("저장된 이미지가 없습니다.")