import numpy as np
import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt # import matplotlib.pyplot as plt


#from sklearn.decomposition import PCA



from sklearn.model_selection import train_test_split 
#from sklearn.dummy import DummyClassifier 


from sklearn.metrics import accuracy_score

# файлы отличаются соотношением включенности двух сенсоров, пока берем один файл.
# внутри файла есть набор данных для разных соотношений спирта и воздуха, и указывается какой именно спирт
# то есть надо пять колонок спиртов заместить на одну колонку с названием спирта? Дописать сзади и удалить эти?

alcohols = ["1-Octanol", "1-Propanol", "2-Butanol", "2-propanol", "1-isobutanol"]
# концентрация привязывается к названию колонок
concentrations = {'0.799_0.201':0.200, '0.799_0.201.1':0.200, '0.700_0.300':0.300, '0.700_0.300.1':0.300, '0.600_0.400':0.400, '0.600_0.400.1':0.400, '0.501_0.499':0.500, '0.501_0.499.1':0.500, '0.400_0.600':0.600, '0.400_0.600.1':0.600}

# Sensor name MIP ratio NP ratio -> chapter of MIP MIP/NP
# QCM3 1 1 => 0.5
# QCM6 1 0 => 1.0
# QCM7 1 0.5 => 0.67
# QCM10 1 2 => 0.33
# QCM12 0 1 => 0.0

# Данные по сенсорам хранятся в словаре где ключ - имя файла, значение - тип сенсора
# https://stackoverflow.com/questions/36244380/enumerate-for-dictionary-in-python 
sensors = {'QCM3': 0.5, 'QCM6': 1.0, 'QCM7': 0.67, 'QCM10': 0.33, 'QCM12': 0.0}


# Data load =========================

df_pairs_columns = ['Concentration', 'Alcohol', 'QCM3', 'QCM6', 'QCM7', 'QCM10', 'QCM12']
df_pairs = pd.DataFrame(columns = df_pairs_columns, index = [i for i in range(250)])

df_columns = ['Concentration', 'Readings', '1-Octanol', '1-Propanol', '2-Butanol', '2-propanol', '1-isobutanol', 'Alcohol']

for sens_name in sensors:
    df = pd.read_csv('./data/' + sens_name + '.csv', sep=';')
    df_format = pd.DataFrame(columns = df_columns, index = [i for i in range(0, 2 * len(df.index))])

    for i in range(df.shape[0]):
        for j in alcohols:
            for k in range(10):
                df_format.at[i*10 + k, j] = df.at[i,j]
                if df.at[i, j] == 1:
                    df_format.at[i*10 + k, "Alcohol"] = j

        for index, (conc_name, conc) in enumerate(concentrations.items()):
            df_format.at[i*10 + index, "Concentration"] = conc
            df_format.at[i*10 + index, "Readings"] = df.at[i, conc_name]
    
    df_pairs[sens_name] = df_format['Readings']
                
    if sens_name == 'QCM3':
        df_pairs['Concentration'] = df_format['Concentration']
        df_pairs['Alcohol'] = df_format['Alcohol']


   
# df_pairs.to_csv('./output/' + 'df_pairs_columns.csv')


df_pairs['Concentration'] = pd.to_numeric(df_pairs['Concentration'], errors='ignore')
df_pairs['QCM3'] = pd.to_numeric(df_pairs['QCM3'], errors='ignore')
df_pairs['QCM6'] = pd.to_numeric(df_pairs['QCM6'], errors='ignore')
df_pairs['QCM7'] = pd.to_numeric(df_pairs['QCM7'], errors='ignore')
df_pairs['QCM10'] = pd.to_numeric(df_pairs['QCM10'], errors='ignore')
df_pairs['QCM12'] = pd.to_numeric(df_pairs['QCM12'], errors='ignore')



'''
# Correlation matrix  =-=-=-==========================
num_cols = df_pairs.select_dtypes(exclude='object')
plt.figure(figsize=(10,10))
sns.heatmap(num_cols.corr(), cmap="RdYlBu_r");
plt.title('Correlation matrix')
plt.show()
'''

'''
# Pair plot ===============
sns.pairplot(df_pairs, vars=['QCM3', 'QCM6', 'QCM7', 'QCM10', 'QCM12'], hue='Alcohol')  
plt.show()
'''

# можно попробовать-таки обучить на примере с максимальным разбросом.  12 и 7
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # "pip install scikit-learn", not "pip install sklearn"
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV


precision = []


pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)), # increased from default (100) https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter
    ]
)

y =  list(df_pairs["Alcohol"])
X = df_pairs.iloc[:, [4, 6]]


min_max_scaler = preprocessing.MinMaxScaler() 
X = min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2) # # Для воспроизводимости вы должны установить аргумент random_state.

model = pipeline.fit(X_train, y_train)
y_pred_simple = model.predict(X_test)
print(classification_report(y_test, y_pred_simple, zero_division=0))


# Оптимизация гиперпараметров

parameters = {
        'scaler__with_mean': [True, False],
        'clf__C': np.linspace(0.01, 1, 10),
        'clf__penalty': ['l2'], #  было 'clf__penalty': ['l2', None] - надоели предупреждения "Setting penalty=None will ignore the C and l1_ratio parameters" 
        'clf__random_state': [2023],
}

random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=10,
        random_state=2023,
        n_jobs=4,
        verbose=1,
)

random_search.fit(X_train, y_train)

    
best_parameters = random_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print(f"{param_name}: {best_parameters[param_name]}")
        
y_pred_optimized = random_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred_optimized, zero_division=0)) # Постоянно ошибки: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.

'''
              precision    recall  f1-score   support

   1-Octanol       0.63      0.92      0.75        13
  1-Propanol       0.00      0.00      0.00         7
1-isobutanol       0.57      0.44      0.50         9
   2-Butanol       0.60      0.30      0.40        10
  2-propanol       0.50      0.45      0.48        11

    accuracy                           0.48        50
   macro avg       0.46      0.42      0.43        50
weighted avg       0.50      0.48      0.47        50
'''

# prediction: