# PYTHON_SKLEARN
Any py. practices related to sklearn objects 


'''
機器學習基礎之 資料預處理（sklearn preprocessing）
Standardization即標準化，儘量將資料轉化為均值為零，方差為一的資料，形如標準正態分佈（高斯分佈）。
實際中我們會忽略資料的分佈情況，僅僅是通過改變均值來集中資料，然後將非連續特徵除以他們的標準差。

### 標準化，均值去除 和 按方差比例縮放 （Standardization, or mean removal and variance scaling）
資料集的標準化：當個體特徵太過或明顯不遵從高斯正態分佈時，標準化表現的效果較差。
實際操作中，經常忽略特徵資料的分佈形狀，移除每個特徵均值，劃分離散特徵的標準差，從而等級化，進而實現資料中心化。
'''
from sklearn import preprocessing
import numpy as np

'''
[ 資料 標準化/歸一化 normalization ]
sklearn中 scale函式提供了簡單快速的 singlearray-like 資料集操作。
scale函式標準化  --  preprocessing.scale(X)

def scale(X, axis=0, with_mean=True, with_std=True, copy=True)
注意，scikit-learn中 assume that all features are centered around zero and have variance in the same order.
公式：(X-X_mean)/X_std 分別計算每個屬性 features(axis=0 列)，如果資料不是這樣的要注意！

引數解釋：
    X：{array-like, sparse matrix} 陣列或者矩陣，一維的資料都可以（但是在0.19版本後一維的資料會報錯了！）
    axis：int型別，初始值為0，axis用來計算均值 means 和標準方差 standard deviations.
        如果是0，則單獨的標準化每個特徵（列），如果是1，則標準化每個觀測樣本（行）。
    with_mean: boolean型別，預設為True，表示將資料均值規範到0
    with_std: boolean型別，預設為True，表示將資料方差規範到1

這種標準化相當於 z-score 標準化(zero-mean normalization)
'''
X = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])

X_scaled = preprocessing.scale(X)         # scale 零均值 單位方差
print('sklearn.preprocessing.scale 標準化  \n', X_scaled)
# cn = preprocessing.scale([[p] for _, _, p in X_scaled]).reshape(-1)
cn = preprocessing.scale([[p[-1]] for p in X_scaled]).reshape(-1)
''' 每個特徵（列），每個觀測樣本（行） '''
print('取標準化後每個樣本里的最後一個值\n', cn)
print('二維轉一維\n', X_scaled.copy().reshape(-1))
'''對於一維資料的一種可能的處理：先轉換成二維，再在結果中轉換為一維'''
print('轉換後的資料零均值（zero mean，均值為零） \n', X_scaled.mean(axis=0))
print('轉換後的資料單位方差（unit variance，方差為1） \n', X_scaled.std(axis=0))

'''
sklearn.preprocessing.StandardScaler()
可儲存訓練集的標準化引數(均值、方差)，然後應用在轉換測試集資料。
一般我們的標準化先在訓練集上進行，在測試集上也應該做同樣 mean 和 variance 的標準化，
'''
scaler = preprocessing.StandardScaler().fit(X)
print('訓練集的標準化引數(均值、方差) \n 均值：', scaler.mean_ , '\n 方差：', scaler.var_)     # 預設為 列
print('將訓練集得到的標準化引數應用到測試集上 \n', scaler.transform([[-1., 1., 0.]]))

'''
注 ：
1）若設定 with_mean=False 或 with_std=False，則不做 centering 或 scaling 處理。
2）scale 和 StandardScaler 可以用於 迴歸模型 中的目標值處理。


二、將資料特徵縮放至某一範圍(scaling features to a range)
另外一種標準化方法是將資料縮放至給定的最小值與最大值之間，通常是０與１之間，可用 MinMaxScaler 實現。
或者 將最大的絕對值縮放至單位大小，可用 MaxAbsScaler 實現。

使用這種標準化方法的原因是，有時資料集的標準差非常非常小，有時資料中有很多很多零（稀疏資料）需要儲存住０元素。

1. MinMaxScaler(最小最大值標準化)
將一個特徵中最大的值轉換為1，最小的那個值轉換為0，其餘的值按照一定比例分佈在（0,1）之間
公式：MinMaxScaler =((X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))) * (max - min) + min

注意，如果原來是稀疏矩陣，因為原來的0轉換後會不在是0，而只有最小值才是0，所以轉換後價格形成一個密集矩陣。
'''

'''例子：將資料縮放至[0, 1]間。訓練過程: fit_transform()'''
X_train = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])
min_max_scaler = preprocessing.MinMaxScaler()

X_train_minmax = min_max_scaler.fit_transform(X_train)
print('將資料縮放至[0, 1]間 \n',X_train_minmax)

'''將上述得到的scale引數應用至測試資料'''
X_test = np.array([[-3., -1., 4.]])
X_test_minmax = min_max_scaler.transform(X_test)
print('將上述得到的scale引數應用至測試資料 \n',X_test_minmax)
print('檢視scaler的屬性 min_max_scaler.scale_  \n', min_max_scaler.scale_)
print('檢視scaler的屬性 min_max_scaler.min_  \n', min_max_scaler.min_)

'''
MinMaxScaler函式：將特徵的取值縮小到一個範圍（如0到1）
將屬性縮放到一個指定的最大值和最小值(通常是1-0)之間，這可以通過 preprocessing.MinMaxScaler 類來實現。
使用這種方法的目的包括：
    1、對於方差非常小的屬性可以增強其穩定性；
    2、維持稀疏矩陣中為0的條目。

有大量異常值的歸一化
sklearn.preprocessing.robust_scale(X, axis=0, with_centering=True,
with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
'''

'''
自定義歸一化函式：大於某個 閾值THRESHOLD 時其屬於1的概率值要大於0.5，小於 THRESHOLD 時概率值小於0.5，
接近最大值時其概率值越接近1，接近最小值時其概率值越接近0。相當於min-max歸一化的一點改進吧。
'''
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def scalerFunc(x, maxv, minv, THRESHOLD=200):
    label = x >= THRESHOLD
    result = 0.5 * (1 + (x - THRESHOLD) * (label / (maxv - THRESHOLD) + (label - 1) / (minv - THRESHOLD)))
    return result

x = np.array([100, 150, 201, 250, 300]).reshape(-1, 1)
scaler = FunctionTransformer(func=scalerFunc, kw_args={'maxv': x.max(), 'minv': x.min()}).fit(x)
print('自定義歸一化函式\n',scaler.transform(x))

'''

2. MaxAbsScaler（絕對值最大標準化）
與上述標準化方法相似，但是它通過除以最大值(特徵值中的最大絕對數)將訓練集縮放至[-1,1]。
這樣的做法並不會改變原來為0的值，所以也不會改變稀疏性。
'''
X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])
max_abs_scaler = preprocessing.MaxAbsScaler()

X_train_maxabs = max_abs_scaler.fit_transform(X_train)
print('X_train_maxabs \n', X_train_maxabs)
print('訓練集每一列的最大絕對值 \n', max_abs_scaler.scale_)

X_test = np.array([[-3., -1., 4.]])
X_test_maxabs = max_abs_scaler.transform(X_test)
print('測試集資料除以訓練集得到的最大絕對值縮放 \n', X_test_maxabs)
