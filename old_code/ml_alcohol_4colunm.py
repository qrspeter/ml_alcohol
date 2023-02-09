import numpy as np
import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt # import matplotlib.pyplot as plt

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


# Data load =========================

df = pd.read_csv(files[0], sep=';') # без разделителя имена не распознаются, хотя числа - распознаются.
print(df.head())


# make new table and fill it

df_full = pd.DataFrame(columns = ['Concentration', 'Sensor', 'Alcohol', 'Readings'], index = [i for i in range(0, file_size*len(files))])


for file_index, file_item in enumerate(files):
    df = pd.read_csv(files[file_index], sep=';')

    for i in range(file_size):
        df_full.at[file_index*file_size + i,"Sensor"] = sensors[file_index]
        
    for i in range(df.shape[0]):
        for j in alcohols:
            if df.at[i, j] == 1:   
                for k in range(10):
                    df_full.at[file_index*file_size + i*10 + k, "Alcohol"] = j
        for index, item in enumerate(conc_columns):
            df_full.at[file_index*file_size + i*10 + index, "Concentration"] = concentrarions[index // 2]
            df_full.at[file_index*file_size + i*10 + index, "Readings"] = df.at[i, conc_columns[index]]

   
   
# или другая таблица   
df_alco = pd.DataFrame(columns = ['Concentration', 'Sensor', 'Readings','1-Octanol', '1-Propanol', '2-Butanol', '2-propanol', '1-isobutanol'], index = [i for i in range(0, file_size*len(files))])

for file_index, file_item in enumerate(files):
    df = pd.read_csv(files[file_index], sep=';')

    for i in range(file_size):
        df_alco.at[file_index*file_size + i,"Sensor"] = sensors[file_index]

    for i in range(df.shape[0]):
        for j in alcohols:
            for k in range(10):
                df_alco.at[file_index*file_size + i*10 + k, j] = df.at[i,j]

        for index, item in enumerate(conc_columns):
            df_alco.at[file_index*file_size + i*10 + index, "Concentration"] = concentrarions[index // 2]
            df_alco.at[file_index*file_size + i*10 + index, "Readings"] = df.at[i, conc_columns[index]]


                
        
print(df_full)


print(df_alco)


df_full.to_csv('df_alco.csv')

# try  Scatterplots with targets =-=-=-==========================

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
for idx, feature in enumerate(df_full.columns):
    df_full.plot(feature, "Readings", subplots=True, kind="scatter", ax=axes[idx // 4, idx % 4])
plt.show()


# Correlation matrix  =-=-=-==========================

df_full['Concentration'] = pd.to_numeric(df_full['Concentration'], errors='ignore') # Using errors=’coerce’. It will replace all non-numeric values with NaN
df_full['Sensor'] = pd.to_numeric(df_full['Sensor'], errors='ignore')
df_full['Readings'] = pd.to_numeric(df_full['Readings'], errors='ignore')

df_full['1-Octanol'] = pd.to_numeric(df_full['1-Octanol'], errors='ignore')
df_full['1-Propanol'] = pd.to_numeric(df_full['1-Propanol'], errors='ignore')
df_full['2-Butanol'] = pd.to_numeric(df_full['2-Butanol'], errors='ignore')
df_full['2-propanol'] = pd.to_numeric(df_full['2-propanol'], errors='ignore')
df_full['1-isobutanol'] = pd.to_numeric(df_full['1-isobutanol'], errors='ignore')


num_cols = df_alco.select_dtypes(exclude='object')

print("num_cols.shape = ", num_cols.shape) # num_cols.shape =  (1250, 0)
print("df_full.dtypes = ", df_alco.dtypes)


plt.figure(figsize=(10,10))
sns.heatmap(num_cols.corr(), cmap="RdYlBu_r");

plt.show()