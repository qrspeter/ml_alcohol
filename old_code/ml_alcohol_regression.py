import numpy as np
import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt # import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler # "pip install scikit-learn", not "pip install sklearn"
from sklearn.decomposition import PCA



from sklearn.model_selection import train_test_split 
from sklearn.dummy import DummyClassifier 
from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

# файлы отличаются соотношением включенности двух сенсоров, пока берем один файл.
# внутри файла есть набор данных для разных соотношений спирта и воздуха, и указывается какой именно спирт
# то есть надо пять колонок спиртов заместить на одну колонку с названием спирта? Дописать сзади и удалить эти?

alcohols = ["1-Octanol", "1-Propanol", "2-Butanol", "2-propanol", "1-isobutanol"]
concentrarions = [0.200, 0.300, 0.400, 0.500, 0.600]
conc_columns = ['0.799_0.201', '0.799_0.201.1', '0.700_0.300', '0.700_0.300.1', '0.600_0.400', '0.600_0.400.1', '0.501_0.499', '0.501_0.499.1', '0.400_0.600', '0.400_0.600.1'] 

# Сенсоры тоже надо перевести в цифру:
# Sensor name MIP ratio NP ratio -> chapter of MIP MIP/NP
# QCM3 1 1 => 0.5
# QCM6 1 0 => 1.0
# QCM7 1 0.5 => 0.67
# QCM10 1 2 => 0.33
# QCM12 0 1 => 0.0
sensors = [0.5, 1.0, 0.67, 0.33, 0.0]

files = ['QCM3.csv', 'QCM6.csv', 'QCM7.csv', 'QCM10.csv', 'QCM12.csv']
file_size = 250 # unsafe if lenght of files differs

#  берем таблицу (или сразу имя файла?), тип сенсора (а тогда уж и какие-то другие параметры...), и возвращаем в формате "концентрация газа, тип сенсора, тип спирта, показания датчика."
def convert_csv():

    return(0)



def draw_3d(data):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    for f in data:
        ax.scatter(f[0], f[1], f[2], c='b')
    plt.show()



# Data load =========================

df = pd.read_csv(files[0], sep=';') # без разделителя имена не распознаются, хотя числа - распознаются.
print(df.head())


# make new table and fill it 
   # минусы - нечисловые объекты "спирты"

# df_full = pd.DataFrame(columns = ['Concentration', 'Sensor', 'Alcohol', 'Readings'], index = [i for i in range(0, file_size*len(files))])


# for file_index, file_item in enumerate(files):
    # df = pd.read_csv(files[file_index], sep=';')

    # for i in range(file_size):
        # df_full.at[file_index*file_size + i,"Sensor"] = sensors[file_index]
        
    # for i in range(df.shape[0]):
        # for j in alcohols:
            # if df.at[i, j] == 1:   
                # for k in range(10):
                    # df_full.at[file_index*file_size + i*10 + k, "Alcohol"] = j
        # for index, item in enumerate(conc_columns):
            # df_full.at[file_index*file_size + i*10 + index, "Concentration"] = concentrarions[index // 2]
            # df_full.at[file_index*file_size + i*10 + index, "Readings"] = df.at[i, conc_columns[index]]

   
# или другая таблица   
df_full = pd.DataFrame(columns = ['Concentration', 'Sensor', 'Readings','1-Octanol', '1-Propanol', '2-Butanol', '2-propanol', '1-isobutanol'], index = [i for i in range(0, file_size*len(files))])

for file_index, file_item in enumerate(files):
    df = pd.read_csv(files[file_index], sep=';')

    for i in range(file_size):
        df_full.at[file_index*file_size + i,"Sensor"] = sensors[file_index]

    for i in range(df.shape[0]):
        for j in alcohols:
            for k in range(10):
                df_full.at[file_index*file_size + i*10 + k, j] = df.at[i,j]

        for index, item in enumerate(conc_columns):
            df_full.at[file_index*file_size + i*10 + index, "Concentration"] = concentrarions[index // 2]
            df_full.at[file_index*file_size + i*10 + index, "Readings"] = df.at[i, conc_columns[index]]



print(df_full)


# <WORKS> 
# df_full.to_csv('df_full2.csv')

# <WORKS> try  Scatterplots with targets =-=-=-==========================

# fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
# for idx, feature in enumerate(df_full.columns):
    # df_full.plot(feature, "Readings", subplots=True, kind="scatter", ax=axes[idx // 4, idx % 4])
# plt.show()



df_full['Concentration'] = pd.to_numeric(df_full['Concentration'], errors='ignore') # Using errors=’coerce’. It will replace all non-numeric values with NaN
df_full['Sensor'] = pd.to_numeric(df_full['Sensor'], errors='ignore')
df_full['Readings'] = pd.to_numeric(df_full['Readings'], errors='ignore')

df_full['1-Octanol'] = pd.to_numeric(df_full['1-Octanol'], errors='ignore')
df_full['1-Propanol'] = pd.to_numeric(df_full['1-Propanol'], errors='ignore')
df_full['2-Butanol'] = pd.to_numeric(df_full['2-Butanol'], errors='ignore')
df_full['2-propanol'] = pd.to_numeric(df_full['2-propanol'], errors='ignore')
df_full['1-isobutanol'] = pd.to_numeric(df_full['1-isobutanol'], errors='ignore')


num_cols = df_full.select_dtypes(exclude='object')

print("num_cols.shape = ", num_cols.shape) # num_cols.shape =  (1250, 0)
print("df_full.dtypes = ", df_full.dtypes)

'''
num_cols.shape =  (1250, 8)
df_full.dtypes =  Concentration    float64
Sensor           float64
Readings         float64
1-Octanol          int64
1-Propanol         int64
2-Butanol          int64
2-propanol         int64
1-isobutanol       int64
dtype: object
'''



# <WORKS> Correlation matrix  =-=-=-==========================

# plt.figure(figsize=(10,10))
# sns.heatmap(num_cols.corr(), cmap="RdYlBu_r");
# plt.show()


#  <WORKS> PCA(n_components=3) ===============================

# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(num_cols)

# model = PCA(n_components=3)
# model.fit(df_scaled)
# df_pca_3 = model.transform(df_scaled)

# draw_3d(df_pca_3)


# <WORKS!> Class visualizations with PCA ===============

# df_class = df_full['Sensor'] # Concentration, , Sensor не уверен что имено надо указать , над пробовать)

# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(num_cols)

# model = PCA(n_components=3)
# model.fit(df_scaled)
# pca_coords = model.transform(df_scaled)

# df_to_draw = pd.DataFrame({
    # 'class': df_class,
    # 'pca1': pca_coords[:, 0],
    # 'pca2': pca_coords[:, 1],
    # 'pca3': pca_coords[:, 2],    
# })
# colors = ['b', 'r', 'k', 'g', 'c', 'm', 'y', 'gold', 'darkorange', 'lime']


# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection='3d')

# for f in df_to_draw.iterrows():
    # ax.scatter(f[1]['pca1'], f[1]['pca2'], f[1]['pca3'], c=colors[int(f[1]['class'])*4]) # for Readings - c=colors[int(f[1]['class'])//50]) is better, for another - smth different from colors[int(f[1]['class'])]
    # # c=colors[int(f[1]['class'])*4] for Sensor, 
# plt.show()


# Learning ========================

# separate alcohols from another? Или просто создать отдельные 1D колонки на разные спирты?

X = df_full.iloc[:, :3]
# y = df_full.iloc[:, 3:]

# одноменных даатфреймов не бывает, нужен просто одномерный массив
#y = pd.DataFrame(columns = ['Alcohol'], index = [i for i in range(0, file_size*len(files))])

# y = np.empty([file_size*len(files), 1], dtype=int)

y = []

#print(y)

for i in range(df_full.shape[0]):
    for index, j  in enumerate(alcohols): #  You can only assign a scalar value not a <class 'str'>
        if df_full.at[i, j] == 1: # if df.at[i, j] == 1
            y.append(j)
 #           y[i, 0] = j #[0, i]
# print(X)
# 

# Increase the number of iterations (max_iter) or scale the data as shown in:
#    https://scikit-learn.org/stable/modules/preprocessing.html
min_max_scaler = preprocessing.MinMaxScaler() 
X = min_max_scaler.fit_transform(X)
#y = min_max_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25) # Для воспроизводимости вы должны установить аргумент random_state.

X_train_minmax = min_max_scaler.fit_transform(X_train)



dummy_clf = DummyClassifier(strategy="most_frequent") 
dummy_clf.fit(X_train, y_train);

y_pred_dummy = dummy_clf.predict(X_test) 
print("accuracy of dummy clf is ", accuracy_score(y_test, y_pred_dummy))




log_reg = LogisticRegression()
log_reg.fit(X_train, y_train); # for np array - y_train.ravel()

y_pred_log_reg = log_reg.predict(X_test) 
print("accuracy of log reg is ", accuracy_score(y_test, y_pred_log_reg))

# accuracy of dummy clf is  0.1853035143769968
# accuracy of log reg is  0.35782747603833864




