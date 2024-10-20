import streamlit as st
import cv2
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import uuid
from tqdm import tqdm

firebase_credentials = json.loads(st.secrets["firebase_credentials"])

   # Khởi tạo Firebase với chứng chỉ từ secrets
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_credentials)
    initialize_app(cred, {'storageBucket': 'hchuong.appspot.com'})

# Kết nối đến Firestore và Storage
db = firestore.client()
bucket = storage.bucket()

# Cấu hình trang
st.set_page_config(layout="wide", page_title="Ứng dụng Tổng hợp")

# CSS để tạo kiểu cho ứng dụng
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .sidebar .sidebar-content {
        width: 300px;
    }
    .table-container {
        display: flex;
        justify-content: center;
        width: 100%;
        overflow-x: auto;
    }
    .dataframe {
        font-size: 14px;
        width: 100%;
        border-collapse: collapse;
    }
    .dataframe th, .dataframe td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    .dataframe td:nth-child(3), .dataframe td:nth-child(4) {
        text-align: center;
    }
    .dataframe img {
        max-width: 80px;
        max-height: 80px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .stApp {
        max-width: 100%;
        margin: 0 auto;
    }
    .search-result {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-top: 10px;
    }
    .search-result img {
        max-width: 100px;
        max-height: 100px;
    }
    .stDataFrame {
        width: 100%;
        max-width: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Các hàm từ file DSSV.py
def upload_image(file):
    if file is not None:
        file_name = str(uuid.uuid4()) + "." + file.name.split(".")[-1]
        blob = bucket.blob(file_name)
        blob.upload_from_file(file)
        blob.make_public()
        return blob.public_url
    return None

def get_student_data():
    students_ref = db.collection("Students")
    students = students_ref.get()
    table_data = []
    for student in students:
        student_data = student.to_dict()
        table_data.append({
            "ID": student.id,
            "Name": student_data.get("Name", ""),
            "TheSV": student_data.get("TheSV", ""),
            "ChanDung": student_data.get("ChanDung", "")
        })
    return table_data

# Các hàm từ file f.py
def init_haar_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar Cascade file not found: {cascade_path}")
    return cv2.CascadeClassifier(cascade_path)

def init_yunet_sface():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yunet_path = os.path.join(current_dir, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(current_dir, "face_recognition_sface_2021dec.onnx")

    if not os.path.exists(yunet_path) or not os.path.exists(sface_path):
        raise FileNotFoundError("YuNet or SFace model file not found")

    face_detector = cv2.FaceDetectorYN.create(yunet_path, "", (0, 0), 0.6, 0.3, 1)
    face_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    
    return face_detector, face_recognizer

def detect_face_haar(image, cascade):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), faces

def detect_recognize_face_yunet(image, face_detector, face_recognizer):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(img)
    
    if faces is not None and len(faces) > 0:
        face = faces[0]
        aligned_face = face_recognizer.alignCrop(img_rgb, face)
        feature = face_recognizer.feature(aligned_face)
        return img_rgb, faces[0], feature
    return img_rgb, None, None

def compare_faces(feature1, feature2, face_recognizer):
    cosine_score = face_recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
    return cosine_score

def draw_faces(img, faces, is_haar=True):
    if is_haar:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        if faces is not None:
            bbox = faces[:4].astype(int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
    return img

# Các hàm từ file cuano.py
@st.cache_resource
def load_models():
    yunet = cv2.FaceDetectorYN.create(
        "face_detection_yunet_2023mar.onnx",
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )
    sface = cv2.FaceRecognizerSF.create(
        "face_recognition_sface_2021dec.onnx",
        ""
    )
    return yunet, sface

yunet, sface = load_models()

def preprocess_face(face_img):
    return cv2.resize(face_img, (112, 112))

def load_known_faces_from_firebase():
    known_faces = []
    known_names = []
    students_ref = db.collection("Students")
    students = students_ref.get()
    for student in tqdm(students, desc="Xử lý ảnh chân dung"):
        student_data = student.to_dict()
        chandung_url = student_data.get("ChanDung")
        if chandung_url:
            try:
                response = requests.get(chandung_url)
                img = Image.open(BytesIO(response.content))
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                face = preprocess_face(img)
                face_feature = sface.feature(face)
                known_faces.append(face_feature)
                known_names.append(student_data.get("Name", "Unknown"))
            except Exception as e:
                st.error(f"Lỗi khi xử lý ảnh của {student_data.get('Name', 'Unknown')}: {e}")
    return known_faces, known_names

def recognize_faces(class_image, known_faces, known_names):
    height, width, _ = class_image.shape
    yunet.setInputSize((width, height))
    _, faces = yunet.detect(class_image)

    recognized_names = set()
    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])
            face_img = class_image[y:y+h, x:x+w]
            face_img = preprocess_face(face_img)
            face_feature = sface.feature(face_img)

            distances = [sface.match(face_feature, known_face, cv2.FaceRecognizerSF_FR_COSINE) for known_face in known_faces]
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]

            if min_distance < 0.3 and known_names[min_distance_index] not in recognized_names:
                name = known_names[min_distance_index]
                recognized_names.add(name)
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(class_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(class_image, f"{name} ({min_distance:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return class_image, recognized_names

# Các hàm chính cho mỗi ứng dụng
def dssv_app():
    st.header("1. Danh sách Sinh viên")
    
    # Khởi tạo session state
    if 'current_action' not in st.session_state:
        st.session_state.current_action = None

    # Các nút chức năng
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Thêm Sinh viên"):
            st.session_state.current_action = 'add'
    with col2:
        if st.button("Tìm kiếm"):
            st.session_state.current_action = 'search'
    with col3:
        if st.button("Hiển thị tất cả"):
            st.session_state.current_action = None

    # Xử lý các chức năng
    if st.session_state.current_action == 'add':
        st.subheader("Thêm Sinh viên Mới")
        new_id = st.text_input("ID Sinh viên")
        new_name = st.text_input("Tên Sinh viên")
        new_thesv = st.file_uploader("Tải lên ảnh thẻ sinh viên", type=["jpg", "png", "jpeg"])
        new_chandung = st.file_uploader("Tải lên ảnh chân dung", type=["jpg", "png", "jpeg"])

        if st.button("Xác nhận thêm"):
            if new_id and new_name and new_thesv and new_chandung:
                thesv_url = upload_image(new_thesv)
                chandung_url = upload_image(new_chandung)
                
                db.collection("Students").document(new_id).set({
                    "Name": new_name,
                    "TheSV": thesv_url,
                    "ChanDung": chandung_url
                })
                st.success("Đã thêm sinh viên mới thành công!")
                st.session_state.current_action = None
                st.experimental_rerun()
            else:
                st.warning("Vui lòng điền đầy đủ thông tin!")

    elif st.session_state.current_action == 'search':
        st.subheader("Tìm kiếm Sinh viên")
        search_id = st.text_input("Nhập ID sinh viên cần tìm")
        if st.button("Xác nhận tìm kiếm"):
            student = db.collection("Students").document(search_id).get()
            if student.exists:
                student_data = student.to_dict()
                st.markdown(f"""
                <div class="search-result">
                    <div>ID: {student.id}</div>
                    <div>Tên: {student_data.get('Name', '')}</div>
                    <img src="{student_data.get('TheSV', '')}" alt="Thẻ Sinh viên">
                    <img src="{student_data.get('ChanDung', '')}" alt="Ảnh Chân dung">
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Không tìm thấy sinh viên với ID này!")

    # Hiển thị bảng dữ liệu với chức năng chỉnh sửa và xóa
    st.subheader("Danh sách Sinh viên")
    table_data = get_student_data()
    df = pd.DataFrame(table_data)

    # Thêm cột cho chức năng chỉnh sửa và xóa
    df['Edit'] = False
    df['Delete'] = False

    edited_df = st.data_editor(
        df,
        hide_index=True,
        column_config={
            "Edit": st.column_config.CheckboxColumn("Chỉnh sửa", default=False, width="small"),
            "Delete": st.column_config.CheckboxColumn("Xóa", default=False, width="small"),
            "TheSV": st.column_config.ImageColumn("Thẻ SV", help="Thẻ sinh viên", width="medium"),
            "ChanDung": st.column_config.ImageColumn("Chân dung", help="Ảnh chân dung", width="medium"),
            "ID": st.column_config.TextColumn("ID", help="ID sinh viên", width="medium"),
            "Name": st.column_config.TextColumn("Tên", help="Tên sinh viên", width="large"),
        },
        disabled=["ID", "Name", "TheSV", "ChanDung"],
        use_container_width=True,
        num_rows="dynamic"
    )

    # Xử lý chỉnh sửa
    students_to_edit = edited_df[edited_df['Edit']]
    if not students_to_edit.empty:
        for _, student in students_to_edit.iterrows():
            st.subheader(f"Chỉnh sửa thông tin cho sinh viên: {student['Name']}")
            edit_id = st.text_input(f"ID mới cho {student['ID']}", value=student['ID'])
            edit_name = st.text_input(f"Tên mới cho {student['ID']}", value=student['Name'])
            edit_thesv = st.file_uploader(f"Thẻ Sinh viên mới cho {student['ID']}", type=["jpg", "png", "jpeg"])
            edit_chandung = st.file_uploader(f"Ảnh Chân dung mới cho {student['ID']}", type=["jpg", "png", "jpeg"])

            if st.button(f"Cập nhật cho {student['ID']}"):
                update_data = {"Name": edit_name}
                if edit_thesv:
                    thesv_url = upload_image(edit_thesv)
                    update_data["TheSV"] = thesv_url
                if edit_chandung:
                    chandung_url = upload_image(edit_chandung)
                    update_data["ChanDung"] = chandung_url
                
                # Xử lý thay đổi ID
                if edit_id != student['ID']:
                    # Lấy dữ liệu hiện tại của sinh viên
                    current_data = db.collection("Students").document(student['ID']).get().to_dict()
                    # Cập nhật dữ liệu hiện tại với dữ liệu mới
                    current_data.update(update_data)
                    # Tạo document mới với ID mới
                    db.collection("Students").document(edit_id).set(current_data)
                    # Xóa document cũ
                    db.collection("Students").document(student['ID']).delete()
                    st.success(f"Đã cập nhật thông tin và ID sinh viên từ {student['ID']} thành {edit_id}!")
                else:
                    # Cập nhật document hiện tại nếu ID không thay đổi
                    db.collection("Students").document(student['ID']).update(update_data)
                    st.success(f"Đã cập nhật thông tin sinh viên {student['ID']}!")
                
                st.experimental_rerun()

    # Xử lý xóa
    students_to_delete = edited_df[edited_df['Delete']]
    if not students_to_delete.empty:
        for _, student in students_to_delete.iterrows():
            st.subheader(f"Xác nhận xóa sinh viên: {student['Name']}")
            if st.button(f"Xác nhận xóa {student['ID']}"):
                db.collection("Students").document(student['ID']).delete()
                st.success(f"Đã xóa sinh viên {student['ID']}!")
                st.experimental_rerun()

def appsen_app():
    st.title("Đánh giá SIFT và ORB trên Synthetic Shapes Dataset")

    # Phần 1: Dataset
    st.header("1. Dataset")
    col1, col2,col3 , col4 , col5 = st.columns(5)
    with col1:
        st.image("synthetic_shapes_dataset/training/image_000003.png", caption="Ảnh 1", use_column_width=True)
        st.image("synthetic_shapes_dataset/training/image_000005.png", caption="Ảnh 3", use_column_width=True)
    with col2:    
        st.image("synthetic_shapes_dataset/training/image_000007.png", caption="Ảnh 5", use_column_width=True)
        st.image("synthetic_shapes_dataset/training/image_000009.png", caption="Ảnh 7", use_column_width=True)
    with col3:
        st.image("synthetic_shapes_dataset/training/image_000011.png", caption="Ảnh 9", use_column_width=True)
        st.image("synthetic_shapes_dataset/training/image_000002.png", caption="Ảnh 2", use_column_width=True)
    with col4 :        
        st.image("synthetic_shapes_dataset/training/image_000004.png", caption="Ảnh 4", use_column_width=True)
        st.image("synthetic_shapes_dataset/training/image_000006.png", caption="Ảnh 6", use_column_width=True)
    with col5:
        st.image("synthetic_shapes_dataset/training/image_000008.png", caption="Ảnh 8", use_column_width=True)
        st.image("synthetic_shapes_dataset/training/image_000010.png", caption="Ảnh 10", use_column_width=True)

    # Phần 2: Giới thiệu SIFT và ORB
    st.header("2. Giới thiệu SIFT và ORB")

    # SIFT
    st.subheader("SIFT (Scale-Invariant Feature Transform)")
    st.write("""
    SIFT là một thuật toán phát hiện và mô tả đặc trưng cục bộ trong hình ảnh, được phát triển bởi David Lowe vào năm 1999. Đặc điểm nổi bật của SIFT là:

    1. Bất biến với tỷ lệ và xoay: SIFT có khả năng phát hiện và mô tả các đặc trưng mà không bị ảnh hưởng bởi sự thay đổi kích thước và góc quay của đối tượng trong ảnh.

    2. Độ ổn định cao: Các đặc trưng SIFT rất ổn định đối với các biến đổi hình ảnh như thay đổi góc nhìn, ánh sáng và nhiễu.

    3. Đặc trưng phong phú: SIFT tạo ra một số lượng lớn các đặc trưng từ hình ảnh, giúp tăng khả năng nhận dạng và so khớp đối tượng.

    4. Mô tả đặc trưng chi tiết: Mỗi đặc trưng SIFT được mô tả bằng một vector 128 chiều, cung cấp thông tin chi tiết về vùng lân cận của điểm đặc trưng.

    SIFT thường được sử dụng trong các ứng dụng như nhận dạng đối tượng, ghép ảnh panorama, và tái tạo 3D.
    """)
    st.image("imagesSen/SIFT-feature-extraction-algorithm-process.png", caption="Minh họa SIFT", use_column_width=True)

    # ORB
    st.subheader("ORB (Oriented FAST and Rotated BRIEF)")
    st.write("""
    ORB là một thuật toán phát hiện và mô tả đặc trưng được phát triển bởi Ethan Rublee và cộng sự vào năm 2011 như một giải pháp thay thế hiệu quả và miễn phí bản quyền cho SIFT và SURF. Đặc điểm chính của ORB bao gồm:

    1. Tốc độ nhanh: ORB được thiết kế để có hiệu suất tính toán cao, cho phép xử lý thời gian thực trên các thiết bị có tài nguyên hạn chế.

    2. Kết hợp hai thuật toán: ORB sử dụng FAST để phát hiện điểm đặc trưng và một phiên bản cải tiến của BRIEF để mô tả đặc trưng.

    3. Bất biến với xoay: ORB cải thiện BRIEF bằng cách thêm tính năng bất biến với xoay, giúp nó hoạt động tốt hơn trong các trường hợp đối tượng bị xoay.

    4. Hiệu quả về bộ nhớ: Mô tả đặc trưng của ORB sử dụng các bit string, giúp tiết kiệm bộ nhớ và tăng tốc độ so khớp.

    5. Miễn phí bản quyền: Không giống như SIFT và SURF, ORB là một thuật toán mã nguồn mở và miễn phí bản quyền.

    ORB thường được sử dụng trong các ứng dụng yêu cầu xử lý thời gian thực như theo dõi đối tượng, ổn định video, và tăng cường thực tế (AR).
    """)

    # Phần 3: Giới thiệu Precision và Recall
    st.header("3. Độ đo Precision và Recall")

    # Precision
    st.subheader("Precision")
    st.write("""
    Precision (độ chính xác) là tỷ lệ các dự đoán đúng trong số các dự đoán tích cực. Trong ngữ cảnh của phát hiện điểm đặc trưng:

    Precision = TP / (TP + FP)

    Trong đó:
    - TP (True Positives): Số điểm đặc trưng được phát hiện chính xác.
    - FP (False Positives): Số điểm được phát hiện là đặc trưng nhưng thực tế không phải.

    Precision cao cho thấy thuật toán có độ chính xác cao trong việc xác định điểm đặc trưng, với ít dự đoán sai tích cực.
    """)
    
    # Recall
    st.subheader("Recall")
    st.write("""
    Recall (độ bao phủ) là tỷ lệ các dự đoán đúng trong số các trường hợp thực sự tích cực. Trong ngữ cảnh của phát hiện điểm đặc trưng:

    Recall = TP / (TP + FN)

    Trong đó:
    - TP (True Positives): Số điểm đặc trưng được phát hiện chính xác.
    - FN (False Negatives): Số điểm đặc trưng thực tế nhưng không được phát hiện.

    Recall cao cho thấy thuật toán có khả năng phát hiện được phần lớn các điểm đặc trưng thực sự trong ảnh.
    """)
    st.image("imagesSen/PR.png", caption="Minh họa Recall", use_column_width=True)

    # Đánh giá chung
    st.write("""
    Trong đánh giá hiệu suất của các thuật toán phát hiện điểm đặc trưng như SIFT và ORB:

    - Precision cao nghĩa là thuật toán có độ tin cậy cao trong việc xác định điểm đặc trưng, giảm thiểu việc phát hiện sai.
    - Recall cao nghĩa là thuật toán có khả năng phát hiện được nhiều điểm đặc trưng quan trọng trong ảnh.

    Thường có sự đánh đổi giữa Precision và Recall. Việc cân bằng giữa hai chỉ số này phụ thuộc vào yêu cầu cụ thể của ứng dụng.
    """)

    # Phần 4: Kết quả thực nghiệm
    st.header("4. Kết quả thực nghiệm")
    col1, col2 = st.columns(2)
    with col1:
        st.image("imagesSen/precision_recall_comparison_line.png", caption="Biểu đồ giá trị Precision và Recall trên tập kiểm thử", use_column_width=True)
    with col2:
        st.image("imagesSen/average_precision_recall_bar.png", caption="Giá trị trung bình tư biểu đồ bên cạnh", use_column_width=True)
    st.write("Kết quả (Xanh là điểm groundtruth, đỏ là điểm phát hiện)")
    col3, col4 = st.columns(2)
    with col3:
        st.image("imagesSen/Sen1.png", caption="Hình 1", use_column_width=True)
    with col4:
        st.image("imagesSen/Sen2.png", caption="Hình 2", use_column_width=True)
    with col3:
        st.image("imagesSen/Sen3.png", caption="Hình 3", use_column_width=True)
    with col4:
        st.image("imagesSen/Sen4.png", caption="Hình 4", use_column_width=True)
    # Phần 5: Đánh giá mô hình
    st.header("5. Đánh giá mô hình")
    st.write("""
    Dựa trên kết quả đã thu được, chúng ta có thể đưa ra một số nhận xét về khả năng của SIFT và ORB thông qua hai độ đo Precision và Recall:

    1. So sánh Precision:
       - SIFT có Precision cao hơn (0.1170) so với ORB (0.1049).
       - Điều này cho thấy SIFT có độ chính xác cao hơn một chút trong việc phát hiện điểm đặc trưng. Tuy nhiên, cả hai phương pháp đều có Precision khá thấp, chỉ khoảng 10-11%.

    2. So sánh Recall:
       - SIFT có Recall cao hơn đáng kể (0.4531) so với ORB (0.3115).
       - Điều này chỉ ra rằng SIFT có khả năng phát hiện được nhiều điểm đặc trưng thực sự hơn so với ORB.

    3. Hiệu suất tổng thể:
       - SIFT vượt trội hơn ORB cả về Precision và Recall trên tập dữ liệu này.
       - SIFT có vẻ nhạy hơn và chính xác hơn trong việc phát hiện điểm đặc trưng trên Synthetic Shapes Dataset.

    4. Đánh đổi giữa Precision và Recall:
       - Cả hai phương pháp đều có Recall cao hơn Precision, cho thấy chúng có xu hướng phát hiện nhiều điểm đặc trưng hơn, nhưng không phải tất cả đều chính xác.
       - SIFT có sự cân bằng tốt hơn giữa Precision và Recall so với ORB.

    5. Khả năng áp dụng:
       - Với Recall cao hơn, SIFT có thể phù hợp hơn cho các ứng dụng yêu cầu phát hiện được nhiều điểm đặc trưng, như ghép ảnh panorama hoặc tái tạo 3D.
       - ORB, mặc dù có hiệu suất thấp hơn, vẫn có thể hữu ích trong các ứng dụng yêu cầu tốc độ xử lý nhanh, vì nó thường nhanh hơn SIFT.

    6. Hạn chế:
       - Precision thấp của cả hai phương pháp (dưới 12%) cho thấy chúng gặp khó khăn trong việc phân biệt chính xác điểm đặc trưng thực sự trên Synthetic Shapes Dataset.
       - Điều này có thể do đặc tính của dataset, có thể chứa các hình dạng phức tạp hoặc không điển hình mà các phương pháp truyền thống gặp khó khăn.

    7. Cơ hội cải thiện:
       - Kết quả này cho thấy có nhiều cơ hội để cải thiện hiệu suất phát hiện điểm đặc trưng trên loại dữ liệu này.
       - Các phương pháp học sâu như SuperPoint có thể có tiềm năng vượt trội hơn các phương pháp truyền thống này trên Synthetic Shapes Dataset.

    Tóm lại, mặc dù SIFT thể hiện hiệu suất tốt hơn ORB trên cả hai độ đo, cả hai phương pháp đều có những hạn chế đáng kể khi áp dụng vào Synthetic Shapes Dataset. Điều này nhấn mạnh nhu cầu phát triển các phương pháp mới, có khả năng xử lý tốt hơn các hình dạng tổng hợp và phức tạp.
    """)

def f_app():
    st.title("Ứng dụng So sánh Ảnh Chân dung và Thẻ Sinh viên")

    haar_cascade = init_haar_cascade()
    yunet_detector, sface_recognizer = init_yunet_sface()

    col1, col2 = st.columns(2)

    with col1:
        st.header("Ảnh Chân dung")
        portrait_image = st.file_uploader("Tải lên ảnh chân dung", type=['jpg', 'jpeg', 'png'])

    with col2:
        st.header("Ảnh Thẻ Sinh viên")
        id_image = st.file_uploader("Tải lên ảnh thẻ sinh viên", type=['jpg', 'jpeg', 'png'])

    # Thêm nút kiểm tra
    check_button = st.button("Kiểm tra")

    if portrait_image and id_image and check_button:
        # Xử lý ảnh chân dung với Haar Cascade
        portrait_img, portrait_faces = detect_face_haar(portrait_image, haar_cascade)
        
        # Xử lý ảnh thẻ sinh viên với YuNet và SFace
        id_img, id_face, id_feature = detect_recognize_face_yunet(id_image, yunet_detector, sface_recognizer)

        if len(portrait_faces) > 0 and id_face is not None:
            # Lấy khuôn mặt lớn nhất từ ảnh chân dung
            largest_face = max(portrait_faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Trích xuất đặc trưng từ khuôn mặt chân dung
            portrait_face_img = portrait_img[y:y+h, x:x+w]
            portrait_face_feature = sface_recognizer.feature(cv2.resize(portrait_face_img, (112, 112)))
            
            # So sánh khuôn mặt
            similarity_score = compare_faces(portrait_face_feature, id_feature, sface_recognizer)

            st.header("Kết quả So sánh")
            st.write(f"Độ tương đồng: {similarity_score:.4f}")

            if similarity_score > 0.363:  # Ngưỡng này có thể điều chỉnh
                st.success("Ảnh chân dung và ảnh thẻ sinh viên KHỚP!")
                color = (0, 255, 0)  # Xanh lá
            else:
                st.error("Ảnh chân dung và ảnh thẻ sinh viên KHÔNG KHỚP!")
                color = (0, 0, 255)  # Đỏ

            # Vẽ hình chữ nhật
            portrait_img_with_rect = draw_faces(portrait_img.copy(), [largest_face])
            id_img_with_rect = draw_faces(id_img.copy(), id_face, is_haar=False)

            col1, col2 = st.columns(2)
            with col1:
                st.image(portrait_img_with_rect, caption="Ảnh Chân dung", use_column_width=True)
            with col2:
                st.image(id_img_with_rect, caption="Ảnh Thẻ Sinh viên", use_column_width=True)
        else:
            st.error("Không thể phát hiện khuôn mặt trong một hoặc cả hai ảnh. Vui lòng thử lại với ảnh khác.")
    elif check_button:
        st.warning("Vui lòng tải lên cả ảnh chân dung và ảnh thẻ sinh viên trước khi kiểm tra.")

def cuano_app():
    st.title("Nhận diện khuôn mặt trong ảnh lớp học")

    uploaded_file = st.file_uploader("Chọn ảnh lớp học", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh lớp học đã tải lên", use_column_width=True)

        if st.button("Nhận diện khuôn mặt"):
            with st.spinner("Đang xử lý..."):
                known_faces, known_names = load_known_faces_from_firebase()
                class_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                result_image, recognized_names = recognize_faces(class_image, known_faces, known_names)

            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption="Kết quả nhận diện", use_column_width=True)
            st.success(f"Số lượng khuôn mặt được nhận diện: {len(recognized_names)}")
            st.write("Danh sách sinh viên được nhận diện:")
            for name in recognized_names:
                st.write(f"- {name}")

# Menu chọn ứng dụng
def main():
    st.sidebar.title('Chọn Ứng dụng')
    app_mode = st.sidebar.selectbox("Chọn ứng dụng",
        ["Danh sách Sinh viên", "Đánh giá SIFT và ORB", "So sánh Ảnh Chân dung và Thẻ Sinh viên", "Nhận diện khuôn mặt trong ảnh lớp học"])

    if app_mode == "Danh sách Sinh viên":
        dssv_app()
    elif app_mode == "Đánh giá SIFT và ORB":
        appsen_app()
    elif app_mode == "So sánh Ảnh Chân dung và Thẻ Sinh viên":
        f_app()
    elif app_mode == "Nhận diện khuôn mặt trong ảnh lớp học":
        cuano_app()

if __name__ == "__main__":
    main()
