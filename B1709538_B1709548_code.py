# ==============================================================================
# == @student: Nguyễn Anh Khải - B1709538
# == @student: Nguyễn Đoàn Nhật Minh - B1709548
# ==============================================================================

# ========================== KẾT NỐI COLAB CÙNG KAGGLE =========================
# Kết nối GG Colab với Drive
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content/drive/My Drive/Kaggle"

# Kiểm tra vị trí hiện tại trong Drive
%pwd

# Di chuyển đến thư mục Kaggle
%cd /content/drive/My Drive/Kaggle/

# from google.colab import files
# !pip install -q kaggle

# Tải data của tập dữ liệu về thư mục Kaggle
!kaggle competitions download -c house-prices-advanced-regression-techniques --force

# =============================== ============== ===============================

# Import các thư viện cần thiết
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sb

# =============================== XỬ LÍ TẬP TRAIN ===============================

# Đọc tập dữ liệu train
train = pandas.read_csv('/content/drive/My Drive/Kaggle/train.csv')

# Hiển thị tập train (1460 rows × 81 columns)
# train

# Kết quả: (5 rows × 81 columns)
train.head()

# Kết quả: (1460, 81)
train.shape

train.info()

# Kiểm tra tập dữ liệu có bị thiếu giá trị không
# Kết quả 6965
if train.isnull().sum().sum() > 0:
  print('Số lượng giá trị bị thiếu:', train.isnull().sum().sum())

# Danh sách các cột có dữ liệu bị thiếu cùng với các giá trị thiếu trong tập dữ liệu
print(train.isnull().sum())

# Danh sách tên các cột có dữ liệu bị thiếu trong tập dữ liệu
# ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
#  'BsmtFinType1','BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageYrBlt',
#  'GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
cols_with_missing = [col for col in train.columns if train[col].isnull().sum().any()]

# Thiết lập data để quăng vào DataFrame
data = {}
for col in cols_with_missing:
  data[col] = [round(train[col].isnull().sum(), 0), round((train[col].isnull().sum() / len(train))*100,2)]
# data

# DataFrame chứa danh sách các cột bị thiếu giá trị và số % giá trị bị thiếu ở các cột
pandas.DataFrame(data=data)

# Danh sách các cột có tổng dữ liệu bị thiếu trong cột so với 1460 > 80%
# Kết quả: ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
list_cols_greater80per = []
for col in cols_with_missing:
  if round((train[col].isnull().sum() / 1460)*100,2) >= 80:
    list_cols_greater80per.append(col)
list_cols_greater80per

# Xoá các cột có dữ liệu bị thiếu > 80%
train =  train.drop(list_cols_greater80per, axis=1);

# Kết quả: (1460, 77)
train.shape

# Gía trị lớn nhất của cột SalePrice
# Kết quả: 755000
print(max(train['SalePrice']))

# Gía trị nhỏ nhất của cột SalePrice
# Kết quả: 34900
print(min(train['SalePrice']))

# Lấy trung bình
# Kết quả: 394950
print((max(train['SalePrice'])+min(train['SalePrice']))/2)

# Chọn các căn nhà có giá dưới khoảng 450000 do khoảng trung bình ~400000
train = train[train['SalePrice']<450000]

train.head()

# Drop bỏ cột nhãn SalePrice (1446, 76)
X_train = train.drop(['SalePrice'], axis=1)

# Kết quả (1446, 76)
X_train.shape

# chọn nhãn
y_labels = train['SalePrice']

# Kết quả: (1446,)
y_labels.shape


columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

# Dùng phương pháp thay thế lắp đầy các giá trị dữ liệu bị thiếu bằng các giá trị trung bình
X_train = X_train.apply(lambda x:x.fillna(x.value_counts().index[0]))
X_train = X_train.fillna(X_train['GarageFinish'].value_counts().index[0])
X_train = X_train.fillna(X_train['BsmtQual'].value_counts().index[0])
X_train = X_train.fillna(X_train['GarageType'].value_counts().index[0])
X_train = X_train.fillna(X_train['GarageQual'].value_counts().index[0])
X_train = X_train.fillna(X_train['GarageCond'].value_counts().index[0])
X_train = X_train.fillna(X_train['BsmtCond'].value_counts().index[0])
X_train = X_train.fillna(X_train['BsmtExposure'].value_counts().index[0])
X_train = X_train.fillna(X_train['BsmtFinType1'].value_counts().index[0])
X_train = X_train.fillna(X_train['FireplaceQu'].value_counts().index[0])

# Kết hợp dữ liệu phân loại
X_train = pandas.get_dummies(X_train, columns=['FireplaceQu','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'])

# Dữ liệu sau bị thiếu trong dữ liệu test bị loại bỏ khỏi dữ liệu train
X_train = X_train.drop(['Condition2_RRAe','Exterior2nd_Other','Condition2_RRAn','Condition2_RRNn','HouseStyle_2.5Fin','RoofMatl_ClyTile','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','Exterior1st_ImStucc','Heating_Floor','Heating_OthW','Electrical_Mix','GarageQual_Ex', 'Exterior1st_Stone','Utilities_NoSeWa'], axis=1)

# =============================== XỬ LÍ TẬP TEST ===============================

# Đọc dữ liệu tập test
test = pandas.read_csv('/content/drive/My Drive/Kaggle/test.csv')

# Kết quả (1459,76)
test.shape

# Xử lí loại bỏ các cột như trên
test = test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)

# Kết quả (1459,76)
test.shape

# Dùng phương pháp thay thế lắp đầy các giá trị dữ liệu bị thiếu bằng các giá trị trung bình cho tập test
test = test.apply(lambda x:x.fillna(x.value_counts().index[0]))
test = test.fillna(test['GarageFinish'].value_counts().index[0])
test = test.fillna(test['BsmtQual'].value_counts().index[0])
test = test.fillna(test['FireplaceQu'].value_counts().index[0])
test = test.fillna(test['GarageType'].value_counts().index[0])
test = test.fillna(test['GarageQual'].value_counts().index[0])
test = test.fillna(test['GarageCond'].value_counts().index[0])
test = test.fillna(test['GarageFinish'].value_counts().index[0])
test = test.fillna(test['BsmtCond'].value_counts().index[0])
test = test.fillna(test['BsmtExposure'].value_counts().index[0])
test = test.fillna(test['BsmtFinType1'].value_counts().index[0])
test = test.fillna(test['BsmtFinType2'].value_counts().index[0])
test = test.fillna(test['BsmtUnfSF'].value_counts().index[0])

# Kết hợp dữ liệu phân loại cho tập test
test = pandas.get_dummies(test, columns=['FireplaceQu','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'])

# Kết quả (1459, 260)
test.shape

# Loại bỏ cột Id
X_test = test.drop(['Id'], axis=1)

# Kết quả (1459, 259)
X_test.shape

# Tham khảo tài liệu thì được hướng dẫn dùng XGBoost - regressor
# Phân loại mô hình
from xgboost import XGBRegressor
xgb_clf = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgb_clf.fit(X_train, Y_labels)

# Độ chính xác sau khi kiểm tra chéo
# Kết quả: 0.911240390855695
from sklearn.model_selection import cross_val_score
xgb_clf_cv = cross_val_score(xgb_clf, X_train, Y_labels, cv=10, )
print(xgb_clf_cv.mean())

# Xây dựng mô hình
xgb_clf = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgb_clf.fit(X_train, Y_labels)

# Kết quả: array([124331.93, 162424.52, 184643.12, ..., 142915.52, 120918.39, 225291.92], dtype=float32)
xgb_test = xgb_clf.predict(test)
xgb_test

# =============== TẠO FILE SUBMIT.CSV ĐỂ SUBMIT LÊN KAGGLE ====================

submission = pandas.DataFrame({
        "Id": test["Id"],
        "SalePrice": xgb_test
    })

submission.to_csv("submit.csv", index=False)
