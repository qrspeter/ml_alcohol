import numpy as np
import pandas as pd

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

#  берем таблицу (или сразу имя файла?), тип сенсора (а тогда уж и какие-то другие параметры...), и возвращаем в формате "концентрация газа, тип сенсора, тип спирта, показания датчика."
def convert_csv():

    return(0)


# Data load =========================

df = pd.read_csv('QCM3.csv', sep=';') # без разделителя имена не распознаются, хотя числа - распознаются.
print(df.head())


# make new table
# по строке на каждый эксперимент (а то непонятно что от чего считать, если концентрация в заголовке?):
# концентрация газа, тип сенсора, тип спирта, показания датчика.
# или делать многомерную таблицу? Вписывая концентрацию в нвоые измерения...

df_3 = pd.DataFrame(columns = ['Concentration', 'Sensor', 'Alcohol', 'Readings'], index = [i for i in range(0, 250)])

   
for i in range(df_3.shape[0]):
    df_3.at[i,"Sensor"] = sensors[0]
    
for i in range(df.shape[0]):
    for j in alcohols:
        if df.at[i, j] == 1:   
            for k in range(10):
                df_3.at[i*10 + k, "Alcohol"] = j
    for index, item in enumerate(conc_columns):
        df_3.at[i*10 + index, "Concentration"] = concentrarions[index // 2]
        df_3.at[i*10 + index, "Readings"] = df.at[i, conc_columns[index]]

   

print(df_3)




'''
merge() for combining data on common columns or indices
With merging, you can expect the resulting dataset to have rows from the parent datasets mixed in together, often based on some commonality.
Depending on the type of merge, you might also lose rows that don’t have matches in the other dataset. 
With concatenation, your datasets are just stitched together along an axis — either the row axis or column axis. Visually, a concatenation with no parameters along rows would look like this
concatenated = pandas.concat([df1, df2])
'''

# потом надо аналогично экспортировать другие файлы и добавить как продолжение таблицы. Дополнительно добавить колонку с указанием типа сенсора.










# Merge data into the one table =========================




# Deleting 5 empty columns ===============


# for i in alcohols:
    # df = df.drop(i, axis=1)
            



#print(df.head())


# или пока ничего не делать, а просто вывести графически?


# или колонки должны быть такими, из расчета по строке на каждый эксперимент (а то непонятно что от чего считать, если концентрация в заголовке?):
# концентрация газа, тип сенсора, тип спирта, показания датчика.
# тогда надо из импортированной таблицы формировать новую.


# try  Scatterplots with targets:

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
for idx, feature in enumerate(df_3.columns):
    df_3.plot(feature, "Alcohol", subplots=True, kind="scatter", ax=axes[idx // 2, idx % 2])
plt.show()