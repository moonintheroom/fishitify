import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2
import json

st.set_page_config(
    page_title="Fishitify",  # 웹 브라우저 탭 제목
    page_icon="🐟",         # 탭 아이콘 (이모지 또는 URL)
    layout="wide",      # 페이지 레이아웃 (centered, wide)
)

st.markdown(
    """
    <style>
    .centered-content {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center; /* 텍스트 중앙 정렬 */
        height: 50px; /* 높이 설정 (조정 가능) */
    }
    .centered-content h1 {
        color: #79ABFF; /* h1 색상 설정 (파란색) */
        font-size: 2.5rem; /* 글꼴 크기 조정 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 중앙 정렬된 콘텐츠 표시
st.markdown(
    """
    <div class="centered-content">
        <h1>Fishitify 방문을 환영합니다 🐳</h1>
        <h3>알고 싶은 해양생물 사진을 업로드 하여 정보를 확인하세요.</h3>
    </div>
    """,
    unsafe_allow_html=True
)
# 클래스 이름 및 고유 색상 설정
class_names = [
    'bang-eo', 'daegu', 'gamseongdom',
    'gasung-eo', 'godeung-eo', 'hwang-eo',
    'nong-eo', 'sung-eo'
]
class_colors = {
    'bang-eo': (0, 0, 255),     # 빨간색
    'daegu': (0, 255, 0),       # 초록색
    'gamseongdom': (255, 0, 0), # 파란색
    'gasung-eo': (0, 255, 255), # 노란색
    'godeung-eo': (255, 105, 180),# 핑크색
    'hwang-eo': (255, 0, 255),  # 자주색
    'nong-eo': (128, 0, 128),   # 보라색
    'sung-eo': (0, 128, 255)    # 주황색
}

# JSON 파일 로드
@st.cache_resource
def load_class_info(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

class_info = load_class_info("fish_class_info.json")  # JSON 파일 경로

# YOLO 모델 로드 함수
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

# 객체 탐지 함수
def detect_objects_yolo(model, image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_array = np.array(image)
    results = model.predict(source=image_array, save=False)
    return results

# Bounding Box를 이미지에 그리는 함수
def draw_boxes(image, results, class_names, class_colors):
    for result in results[0].boxes.data.tolist():  # 첫 번째 이미지 결과 가져오기
        x_min, y_min, x_max, y_max, confidence, class_id = result
        class_name = class_names[int(class_id)]  # 클래스 이름
        color = class_colors[class_name]  # 클래스별 색상

        # OpenCV가 사용하는 BGR 형식으로 변환
        color_bgr = (color[2], color[1], color[0])  # RGB -> BGR

        label = f"{class_name} ({confidence:.2f})"

        # Bounding Box 그리기
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color_bgr, thickness=3)
        cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

    return image

def resize_with_aspect_ratio(image, target_width, target_height):
    """
    비율을 유지하면서 이미지를 지정된 크기에 맞게 확장하거나 축소합니다.
    """
    # RGBA 이미지를 RGB로 변환
    if image.shape[2] == 4:  # 4채널 이미지일 경우
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # 원본 크기와 비율 계산
    height, width, _ = image.shape
    aspect_ratio = width / height

    # 새 크기 계산
    if width / target_width > height / target_height:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    # 이미지 크기 조정
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 목표 크기의 캔버스 생성 (흰색 배경)
    canvas = np.full((target_height, target_width, 3), 1500, dtype=np.uint8)

    # 이미지 중앙 정렬
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return canvas

# 모델 로드 여부 확인
if "model" not in st.session_state:
    model_path = "best.pt"
    
    if Path(model_path).exists():
        st.session_state.model = load_yolo_model(model_path)
    else:
        st.warning("유효한 YOLO 모델 경로를 입력하세요.")

# 모델 로드 후 객체 탐지 기능
if "model" in st.session_state:
    model = st.session_state.model

    # 이미지 업로드
    st.markdown(
        """
        <style>
        div[data-testid="stFileUploader"] {
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 18px; /* 글자 크기 */
            max-width: 1000px; /* 최대 너비 설정 */
            margin: 20px auto; /* 가운데 정렬 및 간격 */
            padding: 10px;
            border: 2px dashed #6c757d; /* 테두리 스타일 */
            border-radius: 10px; /* 둥근 모서리 */
            background-color: #f8f9fa; /* 배경색 */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Streamlit 업로더 사용
    uploaded_file = st.file_uploader("이미지를 업로드하세요.", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        # YOLO 모델로 객체 탐지
        results = detect_objects_yolo(model, image)

        # 결과 이미지 생성 및 바운딩 박스 추가
        result_image = np.array(image)
        result_image = draw_boxes(result_image, results, class_names, class_colors)

        # 고정된 크기로 이미지 조정
        max_width, max_height = 800, 600  # 원하는 출력 크기
        result_image = resize_with_aspect_ratio(result_image, max_width, max_height)

        # 왼쪽 칸에 이미지 중앙 정렬
        col1, col2 = st.columns(2)
        with col1:
            # 내부적으로 열을 추가해 중앙 정렬
            inner_col1, inner_col2, inner_col3 = st.columns([1, 2, 1])  # 중앙 열 비율을 크게 설정
            with inner_col2:
                st.image(result_image, caption="객체 탐지 결과", use_container_width=False)

        # 오른쪽 칸에 정보 표시
        with col2:
            st.subheader("객체 탐지 완료 ❗")

            # 표시된 클래스를 추적하기 위한 집합 생성
            displayed_classes = set()

            for result in results[0].boxes.data.tolist():
                x_min, y_min, x_max, y_max, confidence, class_id = result
                class_name = class_names[int(class_id)]

                # 중복된 클래스는 건너뜀
                if class_name in displayed_classes:
                    continue

                displayed_classes.add(class_name)
                match_percentage = round(confidence * 100, 2)  # 소수점 2자리까지 반올림

                # 클래스 정보 가져오기
                info = class_info.get(class_name, {})
                st.write(f"🐟 **{info.get('name', class_name)}** ({match_percentage}% 일치)")
                st.write(f"  - **먹을 수 있나요?**: {info.get('edible', '정보 없음')}")
                st.write(f"  - **종류**: {info.get('type', '정보 없음')}")
                st.write(f"  - **서식지**: {info.get('habitat', '정보 없음')}")
                st.write(f"  - **설명**: {info.get('description', '정보 없음')}")

else:
    st.warning("먼저 YOLO 모델을 로드하세요.")