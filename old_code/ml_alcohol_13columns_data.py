import numpy as np
import pandas as pd

from matplotlib import pyplot as plt # import matplotlib.pyplot as plt


# Data load =========================

df = pd.read_csv('QCM3.csv', sep=';') # без разделителя имена не распознаются, хотя числа - распознаются.
print(df.head())

# замена на концентрацию спирта
df.rename(columns = {'0.799_0.201':'0.200', '0.799_0.201.1':'0.201', '0.700_0.300':'0.300', '0.700_0.300.1':'0.301', '0.600_0.400':'0.400', '0.600_0.400.1':'0.401', '0.501_0.499':' 0.500', '0.501_0.499.1':'0.501', '0.400_0.600':'0.600', '0.400_0.600.1':'0.601'}, inplace = True )


# файлы отличаются соотношением включенности двух сенсоров, пока берем один файл.
# внутри файла есть набор данных для разных соотношений спирта и воздуха, и указывается какой именно спирт
# то есть надо пять колонок спиртов заместить на одну колонку с названием спирта? Дописать сзади и удалить эти?
alcohols = ["1-Octanol", "1-Propanol", "2-Butanol", "2-propanol", "1-isobutanol"]

df.insert(0, "Alcohol", 'x')
# print(df.head())

# сенсоры тоже надо перевести в цифру:
# Sensor name MIP ratio NP ratio -> chapter of MIP MIP/NP
# QCM3 1 1 => 0.5
# QCM6 1 0 => 1.0
# QCM7 1 0.5 => 0.67
# QCM10 1 2 => 0.33
# QCM12 0 1 => 0.0

df.insert(1, "Sensor", 0.5)


# потом надо аналогично экспортировать другие файлы и добавить как продолжение таблицы. ДОполнительно добавить колонку с указанием типа сенсора.



# Merge data into the one column =========================




# Deleting 5 empty columns ===============

for i in range(df.shape[0]):
#    print ('i = ', i)
    for j in alcohols:
#        print(j)
        if df.at[i, j] == 1:
            df.at[i,"Alcohol"] = j

for i in alcohols:
    df = df.drop(i, axis=1)
            

# print(df.head())

print(df)


# или пока ничего не делать, а просто вывести графически?


# или колонки должны быть такими, из расчета по строке на каждый эксперимент (а то непонятно что от чего считать, если концентрация в заголовке?):
# концентрация газа, тип сенсора, тип спирта, показания датчика.
# тогда надо из импортированной таблицы формировать новую.


# try  Scatterplots with targets:

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(25, 20))
for idx, feature in enumerate(df.columns):
    df.plot(feature, "Alcohol", subplots=True, kind="scatter", ax=axes[idx // 4, idx % 4])
plt.show()