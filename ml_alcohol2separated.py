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
#concentrarions = [0.200, 0.300, 0.400, 0.500, 0.600]
#conc_columns = ['0.799_0.201', '0.799_0.201.1', '0.700_0.300', '0.700_0.300.1', '0.600_0.400', '0.600_0.400.1', '0.501_0.499', '0.501_0.499.1', '0.400_0.600', '0.400_0.600.1'] 
concentrations = {'0.799_0.201':0.200, '0.799_0.201.1':0.200, '0.700_0.300':0.300, '0.700_0.300.1':0.300, '0.600_0.400':0.400, '0.600_0.400.1':0.400, '0.501_0.499':0.500, '0.501_0.499.1':0.500, '0.400_0.600':0.600, '0.400_0.600.1':0.600}

# Сенсоры тоже надо перевести в цифру:
# Sensor name MIP ratio NP ratio -> chapter of MIP MIP/NP
# QCM3 1 1 => 0.5
# QCM6 1 0 => 1.0
# QCM7 1 0.5 => 0.67
# QCM10 1 2 => 0.33
# QCM12 0 1 => 0.0

# это надо было в словарь... https://stackoverflow.com/questions/36244380/enumerate-for-dictionary-in-python 
# sensors = [0.5, 1.0, 0.67, 0.33, 0.0]
# files = ['QCM3.csv', 'QCM6.csv', 'QCM7.csv', 'QCM10.csv', 'QCM12.csv']
sensors = {'QCM3': 0.5, 'QCM6': 1.0, 'QCM7': 0.67, 'QCM10': 0.33, 'QCM12': 0.0}

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

data_lst = [] # data_lst.append(j)
df_columns = ['Concentration', 'Readings', 'Sensor', '1-Octanol', '1-Propanol', '2-Butanol', '2-propanol', '1-isobutanol', 'Alcohol']

for sens_name in sensors:
    df = pd.read_csv(sens_name + '.csv', sep=';')
    df_format = pd.DataFrame(columns = df_columns, index = [i for i in range(0, 2 * len(df.index))])

#    for i in range(len(concentrations) * len(df.index)):
#        df_format.at[i,"Sensor"] = sensors[sens_name] 
    
    for i in range(df.shape[0]):
        for j in alcohols:
            for k in range(10):
                df_format.at[i*10 + k, j] = df.at[i,j]
                if df.at[i, j] == 1:
                    df_format.at[i*10 + k, "Alcohol"] = j

        for index, (conc_name, conc) in enumerate(concentrations.items()):
            df_format.at[i*10 + index, "Concentration"] = conc
            df_format.at[i*10 + index, "Readings"] = df.at[i, conc_name]
            
    df_format['Concentration'] = pd.to_numeric(df_format['Concentration'], errors='ignore') # Using errors=’coerce’. It will replace all non-numeric values with NaN
    df_format['Sensor'] = pd.to_numeric(df_format['Sensor'], errors='ignore')
    df_format['Readings'] = pd.to_numeric(df_format['Readings'], errors='ignore')
    df_format['1-Octanol'] = pd.to_numeric(df_format['1-Octanol'], errors='ignore')
    df_format['1-Propanol'] = pd.to_numeric(df_format['1-Propanol'], errors='ignore')
    df_format['2-Butanol'] = pd.to_numeric(df_format['2-Butanol'], errors='ignore')
    df_format['2-propanol'] = pd.to_numeric(df_format['2-propanol'], errors='ignore')
    df_format['1-isobutanol'] = pd.to_numeric(df_format['1-isobutanol'], errors='ignore')
   
    tpl = (sens_name, df_format)
   
    data_lst.append(tpl)
    
   
    
#print("data_lst[0] = \n", data_lst[0])
#print("data_lst[3] = \n", data_lst[3])


# <WORKS> 
#df_full.to_csv('df_full2.csv')

# <WORKS> try  Scatterplots with targets =-=-=-==========================

# fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
# for idx, feature in enumerate(df_full.columns):
    # df_full.plot(feature, "Readings", subplots=True, kind="scatter", ax=axes[idx // 4, idx % 4])
# plt.show()




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



x = np.zeros(len(sensors))
y_dummy = np.zeros(len(sensors))
y_log = np.zeros(len(sensors))
#comparison = np.zeros(shape=(len(sensors), 3))
# Make a loop for files or part of dataframe
# возможно надо будет список собирать из значений при переборе?
#for sens_index, (sens_name, sens_ratio) in enumerate(sensors.items()):
for i, sensor in enumerate(data_lst):
    df = sensor[1]
    sens_name = sensor[0]
    y =  list(df["Alcohol"])
    X = df.iloc[:, :2]

    # Increase the number of iterations (max_iter) or scale the data as shown in:
    #    https://scikit-learn.org/stable/modules/preprocessing.html
    min_max_scaler = preprocessing.MinMaxScaler() 
    X = min_max_scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25) # Для воспроизводимости вы должны установить аргумент random_state.
    
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    
    dummy_clf = DummyClassifier(strategy="most_frequent") 
    dummy_clf.fit(X_train, y_train);
    
    x[i] = sensors[sens_name]
    
    y_pred_dummy = dummy_clf.predict(X_test) 
    acc = accuracy_score(y_test, y_pred_dummy)
    print(sens_name, "accuracy of dummy clf is ", acc)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train); # for np array - y_train.ravel()
    y_dummy[i] = acc

    
    y_pred_log_reg = log_reg.predict(X_test) 
    acc = accuracy_score(y_test, y_pred_log_reg)
    print(sens_name, "accuracy of log reg is ", acc)
    y_log[i] = acc


#comparison[comparison[:, 0].argsort()]


plt.plot(x,y_dummy, 'ko' , label = 'Dummy') 
plt.plot(x,y_log, 'r*' , label = 'Log reg') 
plt.legend(loc = 'center right')

plt.xlabel('Sensor')
plt.ylabel('Accuracy')

plt.show()

'''
QCM3 accuracy of dummy clf is  0.12698412698412698
QCM3 accuracy of log reg is  0.42857142857142855
QCM6 accuracy of dummy clf is  0.12698412698412698
QCM6 accuracy of log reg is  0.4444444444444444
QCM7 accuracy of dummy clf is  0.12698412698412698
QCM7 accuracy of log reg is  0.38095238095238093
QCM10 accuracy of dummy clf is  0.12698412698412698
QCM10 accuracy of log reg is  0.4444444444444444
QCM12 accuracy of dummy clf is  0.12698412698412698
QCM12 accuracy of log reg is  0.3968253968253968
'''



