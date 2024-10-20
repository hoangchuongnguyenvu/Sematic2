import streamlit as st
import numpy as np
from PIL import Image

def main():
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

if __name__ == "__main__":
    main()
