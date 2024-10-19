import pandas as pd
import streamlit as st
import numpy as np
from joblib import load

# Tải các mô hình đã lưu
bagging_model = load('linear_model.joblib')
ridge_bagging_model = load('ridge_model.joblib')
mlp_bagging_model = load('neural_net_model.joblib')

# Tạo giao diện ứng dụng
st.title("Dự đoán giá cổ phiếu Vinamilk")

# Tải dữ liệu
uploaded_file = st.file_uploader("Chọn tệp CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Kiểm tra tên cột
    st.write("Dữ liệu đã tải lên:")
    st.dataframe(data)
    st.write("Các cột trong dữ liệu:", data.columns.tolist())
    
    # Chuyển đổi cột 'Ngày' sang kiểu datetime
    if 'Ngày' in data.columns:
        data['Ngày'] = pd.to_datetime(data['Ngày'], format='%d/%m/%Y', errors='coerce')
    else:
        st.error("Không tìm thấy cột 'Ngày' trong dữ liệu.")

    # Lấy giá trị cột 'Đóng cửa' và chuyển đổi sang số
    if 'Đóng cửa' in data.columns:
        data['Đóng cửa'] = data['Đóng cửa'].str.replace(',', '').astype(float)  # Chuyển đổi kiểu dữ liệu
    else:
        st.error("Không tìm thấy cột 'Đóng cửa' trong dữ liệu.")

    # Thêm các đường trung bình vào DataFrame
    data['MA_5'] = data['Đóng cửa'].rolling(window=5).mean()  # Đường trung bình 5 ngày
    data['MA_10'] = data['Đóng cửa'].rolling(window=10).mean()  # Đường trung bình 10 ngày
    data['MA_20'] = data['Đóng cửa'].rolling(window=20).mean()  # Đường trung bình 20 ngày
    data['MA_50'] = data['Đóng cửa'].rolling(window=50).mean()  # Đường trung bình 50 ngày

    # Xóa các giá trị NaN
    data = data.dropna()

    # Chọn mô hình dự đoán
    model_option = st.selectbox("Chọn mô hình dự đoán:", 
                                ('Linear Regression', 'Ridge Regression', 'Neural Network'))

    # Dự đoán
    if st.button("Dự đoán"):
        # Sử dụng 4 cột đầu vào cho dự đoán
        features = data[['Đóng cửa', 'MA_5', 'MA_10', 'MA_20', 'MA_50']].values[-1].reshape(1, -1)  # Chọn hàng cuối cùng

        # Dự đoán với mô hình đã chọn
        if model_option == 'Linear Regression':
            prediction = bagging_model.predict(features)
        elif model_option == 'Ridge Regression':
            prediction = ridge_bagging_model.predict(features)
        elif model_option == 'Neural Network':
            prediction = mlp_bagging_model.predict(features)
        
        # Hiển thị kết quả dự đoán
        st.write(f"Giá cổ phiếu Vinamilk dự đoán: VNĐ{prediction[0]:,.2f}")
