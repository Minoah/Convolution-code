# [Group 11] Convolutional code
Phạm vi demo, sử dụng ảnh nhị phân để làm ví dụ.

Mô tả cấu trúc
Source code bao gồm:
+ Folder Datasets: lưu các hình ảnh bao gồm ảnh png màu, ảnh png nhị phân và file npy chuyển đổi từ ảnh png nhị phân.
+ File BinaryImageSample: để chuyển đổi ảnh màu sang ảnh nhị phân và chuyển ảnh nhị phân sang dạng file npy để chuyển đổi.
+ File ConvEncoder: Mô tả các hàm mô tả bộ mã hóa và các hàm giả lập quá trình chuyền thông tin trong thực tế với sóng mang và nhiễu
+ File ViterbiDecoder: hàm mô tả bộ giải mã với thuật toán Viterbi cho ví dụ cụ thể là bộ mã hóa (2,1,2) rate = 1/2 và độ dài giới hạn constraint length K = 3, bảng trạng thái như trong hình tại báo cáo.
+ File Demo: Mô tả các giá trị biến thu được với tham số cụ thể của bộ mã (2,1,2) để thu được ảnh decode.

Báo cáo: https://drive.google.com/drive/folders/1mgc0ixLUUGMKYhuShaEW6wsCGkTmiKCC


