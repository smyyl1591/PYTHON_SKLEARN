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


'''
Scikit-learn：資料預處理 Preprocessing data
標準化、資料最大最小縮放處理、正則化、特徵二值化和資料缺失值處理。
Note: 一定要注意歸一化是歸一化什麼，歸一化features還是samples。

資料標準化：去除均值和方差進行縮放 Standardization: mean removal and variance scaling
當單個特徵的樣本取值相差甚大或明顯不遵從高斯正態分佈時，資料標準化表現的效果較差。
實際操作中，經常忽略特徵資料的分佈形狀，移除每個特徵均值，劃分離散特徵的標準差，從而等級化，進而實現資料中心化。
Note: test set、training set 都要做相同的預處理操作（standardization、data transformation、etc）

'''
from sklearn import preprocessing
import numpy as np

'''
二值化 Binarization 主要是為了將 資料特徵 轉變成 boolean變數。
在sklearn中，sklearn.preprocessing.Binarizer 函式可以實現這一功能。
可以設定一個閾值(預設為0.0)，結果資料值大於閾值的為1，小於閾值的為0。
'''
X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
binarizer = preprocessing.Binarizer().fit(X)
print('binarizer 列印函式 \n',binarizer)
print('binarizer.transform(X) 轉換為list \n',binarizer.transform(X))

binarizer22 = preprocessing.Binarizer(copy=True, threshold=1.0).fit(X)
print('binarizer22 列印函式 \n',binarizer22)
print('binarizer22.transform(X) 轉換為list \n',binarizer22.transform(X))

from sklearn import preprocessing
from sklearn import tree

'''
對於標稱型資料來說，preprocessing.LabelBinarizer是一個很好用的工具。
比如可以把yes和no轉化為0和1，或是把incident和normal轉化為0和1。當然，對於兩類以上的標籤也是適用的。
help(preprocessing.LabelBinarizer) # 檢視詳細用法
'''
featureList=[[1,0],[1,1],[0,0],[0,1]]  # 特徵矩陣
labelList=['yes', 'no', 'no', 'yes']  # 標籤矩陣
lb = preprocessing.LabelBinarizer()  # 將標籤矩陣二值化
dummY=lb.fit_transform(labelList)
print('將標籤矩陣二值化 \n',dummY)

# 模型建立和訓練
clf = tree.DecisionTreeClassifier()
clf = clf.fit(featureList, dummY)
#print(clf)
p=clf.predict([[0,1]])
print(p)  # [1]

# 逆過程 再把 0 和 1 轉回原始 數值
yesORno=lb.inverse_transform(p)
print(yesORno)

'''
缺失值處理Imputation of missing values
由於許多現實中的資料集都包含有缺失值，要麼是空白的，要麼使用NaNs或者其它的符號替代。
這些資料無法直接使用 scikit-learn 分類器直接訓練，所以需要進行處理。
幸運地是，sklearn 中的 Imputer 類提供了一些基本的方法來處理缺失值，
如使用均值、中位值或者缺失值所在列中頻繁出現的值來替換。
Imputer類同樣支援稀疏矩陣。

'''
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2], [np.nan, 3], [7, 6]]) # 擬合得出 列均值，填充給 X 的 缺失值

# Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print('缺失值處理Imputation of missing values \n',imp.transform(X))

'''
正則化Normalization 的過程是將每個樣本縮放到單位範數(每個樣本的範數為1)，
如果要使用如二次型(點積)或者其它核方法計算兩個樣本之間的相似性這個方法會很有用。
該方法是文字分類和聚類分析中經常使用的向量空間模型（Vector Space Model)的基礎.
Normalization 主要思想是對每個樣本計算其p-範數，然後對該樣本中每個元素除以該範數，
這樣處理的結果是使得每個處理後樣本的p-範數(l1-norm,l2-norm)等於1。

def normalize(X, norm='l2', axis=1, copy=True)
注意，這個操作是對所有樣本（而不是features）進行的，也就是將每個樣本的值除以這個樣本的Li範數。
所以這個操作是針對axis=1進行的。

三、正則化
對每個樣本計算其 p-範數，再對每個元素除以該範數，這使得每個處理後樣本的p-範數（l1-norm,l2-norm）等於1。
如果後續要使用二次型等方法計算兩個樣本之間的相似性會有用。
preprocessing.Normalizer(norm=’l2’, copy=True)

'''
X = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])
X_normalized = preprocessing.normalize(X, norm='l2')
print( '正則化Normalization\n',X_normalized )
normalizer = preprocessing.Normalizer().fit(X)
print( 'normalizer.transform(X) 正則化 \n' , normalizer.transform(X) )
print( '所有元素求和 \n' , normalizer.transform(X).sum() )
print( 'X所有元素的範數和 \n' , abs(X).sum() )
# print( abs(X).sum(axis=0) )
# print( abs(X).sum(axis=1) )
print( 'X所有元素都除以 所有元素的範數和 \n' , X / X.sum() )

'''
幾個概念
1-範數：向量各分量絕對值之和
2-範數：向量長度
最大範數：向量各分量絕對值的最大值
p-範數的計算公式：||X||p=(|x1|^p+|x2|^p+…+|xn|^p)^1/p
'''

'''
標準化（Scale）和正則化（Normalization）是兩種常用的資料預處理方法，
其作用是讓資料變得更加“規範”一些。在文字聚類等任務中使用的比較多。
1.資料標準化
公式為：(X-mean)/std  計算時對每個屬性/每列分別進行。將資料按期屬性（按列進行）減去其均值，並處以其方差。
得到的結果是，對於每個屬性/每列來說所有資料都聚集在0附近，方差為1。經過資料標準化的資料，可以看到有些特徵被凸顯出來了。

2.資料正則化
正則化的過程是將每個樣本縮放到單位範數（每個樣本的範數為1），
如果後面要使用如二次型（點積）或者其它核方法計算兩個樣本之間的相似性這個方法會很有用。

Normalization主要思想是對每個樣本計算其p-範數，然後對該樣本中每個元素除以該範數，
這樣處理的結果是使得每個處理後樣本的p-範數（l1-norm,l2-norm）等於1。
p-範數的計算公式：||X||p=(|x1|^p+|x2|^p+...+|xn|^p)^1/p
該方法主要應用於文字分類和聚類中。例如，對於兩個TF-IDF向量的l2-norm進行點積，就可以得到這兩個向量的餘弦相似性。

在sklearn中有三種正則化方法，l1範數、l2範數、max範數。
使用這三種範數生成的結果如下圖所示：
在肉眼上很難看出有什麼區別，
不過還是能看出l2範數的結果相對更好，即能儘可能的削弱“強勢”特徵，將一些數值較小但是比較有特點的特徵“凸顯”出來。

sklearn官方文件：
http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale
http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html#sklearn.preprocessing.normalize
'''

'''
更多的資料預處理方法參考官方文件：
http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling

參考官方文件：http://scikit-learn.org/stable/modules/preprocessing.html
官網：http://scikit-learn.org/stable/
'''
