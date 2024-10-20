# -*- coding: utf-8 -*-

import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import uuid

st.set_page_config(layout="wide")

# Khởi tạo Firebase (chỉ thực hiện một lần)
if not firebase_admin._apps:
    cred = credentials.Certificate("hchuong-firebase-adminsdk-1m82k-829fb1690b.json")
    firebase_admin.initialize_app(cred, {'storageBucket': 'hchuong.appspot.com'})

# Kết nối đến Firestore và Storage
db = firestore.client()
bucket = storage.bucket()

# Khởi tạo session state
if 'current_action' not in st.session_state:
    st.session_state.current_action = None

# CSS để tạo kiểu cho ứng dụng
st.markdown("""
<style>
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

# Hàm để tải lên hình ảnh vào Firebase Storage
def upload_image(file):
    if file is not None:
        file_name = str(uuid.uuid4()) + "." + file.name.split(".")[-1]
        blob = bucket.blob(file_name)
        blob.upload_from_file(file)
        blob.make_public()  # Đảm bảo URL có thể truy cập công khai
        return blob.public_url
    return None

# Hàm để lấy dữ liệu sinh viên từ Firestore
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

# Tiêu đề ứng dụng
st.header("1. Danh sách Sinh viên")

# Tạo các nút cho các chức năng
col1, col2 = st.columns(2)
with col1:
    if st.button("Thêm Sinh viên"):
        st.session_state.current_action = 'add'
with col2:
    if st.button("Tìm kiếm Sinh viên"):
        st.session_state.current_action = 'search'

# Chức năng thêm sinh viên mới
if st.session_state.current_action == 'add':
    st.subheader("Thêm Sinh viên mới")
    new_id = st.text_input("ID")
    new_name = st.text_input("Tên")
    new_thesv = st.file_uploader("Thẻ Sinh viên", type=["jpg", "png", "jpeg"])
    new_chandung = st.file_uploader("Ảnh Chân dung", type=["jpg", "png", "jpeg"])

    if st.button("Xác nhận thêm"):
        if new_id and new_name and new_thesv and new_chandung:
            thesv_url = upload_image(new_thesv)
            chandung_url = upload_image(new_chandung)
            db.collection("Students").document(new_id).set({
                "Name": new_name,
                "TheSV": thesv_url,
                "ChanDung": chandung_url
            })
            st.success("Đã thêm sinh viên mới!")
            st.session_state.current_action = None
            st.rerun()  # Làm mới trang để cập nhật dữ liệu
        else:
            st.warning("Vui lòng điền đầy đủ thông tin!")

# Chức năng tìm kiếm
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
            
            st.rerun()

# Xử lý xóa
students_to_delete = edited_df[edited_df['Delete']]
if not students_to_delete.empty:
    for _, student in students_to_delete.iterrows():
        st.subheader(f"Xác nhận xóa sinh viên: {student['Name']}")
        if st.button(f"Xác nhận xóa {student['ID']}"):
            db.collection("Students").document(student['ID']).delete()
            st.success(f"Đã xóa sinh viên {student['ID']}!")
            st.rerun()