import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('winequality-red.csv', sep=';')




rom sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TODO: 1
# проверим на пропуски и на дубликаты
display(data.info())
display(data.duplicated().value_counts())

# удаляем дубликаты
data = data.drop_duplicates()




# TODO: 3
# проанализируем признаки
display(data.describe())

# стандартизируем данные
scaler = StandardScaler(with_std=True)
scaled_data = scaler.fit_transform(data)

# заранее определим стандартизированные данные в pd DataFrame,
# для удобства
scaled_data = pd.DataFrame(data=scaled_data, columns=data.keys())

# построим коробочные графики для определения выбросов
plt.figure(figsize=(25,15))
scaled_data.boxplot()

# удалим выбросы и пропущенные значения
scaled_data = scaled_data[(scaled_data < 4) & (scaled_data > -3)]
data = scaled_data.dropna()

# посмотрим информацию о предобработанных данных
display(data.info())


# TODO: 2
# разделим датасет на тренировочные и нестовые выборки
X = data.drop(['quality'], axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

display(X.shape)



from sklearn.model_selection import cross_val_score

# рассчитываем среднее качество кросс валидации на дефолтных параметрах
cv_tree_result = cross_val_score(DecisionTreeRegressor(random_state=42), X_train, y_train, scoring='r2')

cv_adaBoost_result = cross_val_score(AdaBoostRegressor(random_state=42), X_train, y_train, scoring='r2')

print('Кросс-валидация R2 для DecisionTreeRegressor:', np.mean(cv_tree_result))
print('Кросс-валидация R2 для AdaBoostRegressor', np.mean(cv_adaBoost_result))

from sklearn.model_selection import GridSearchCV

# инициализируем сетки параметров
tree_grid = {
    'splitter' : ['best', 'random'],
    'max_depth' : [None, 5,6,7,8,10,11,12,13,14,15,16,17,18,20],
    'min_samples_split' : [2,3,4,5,6,7,8,9,10],
    'max_features' : ['auto', 'sqrt', 'log2']
}

ab_grid = {
    'n_estimators' : [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200],
    'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 1],
    'loss' : ['linear', 'square', 'exponential']
}

# обучаемся
gs_tree = GridSearchCV(DecisionTreeRegressor(random_state=42), tree_grid, scoring='r2')
gs_tree.fit(X_train, y_train)
best_tree_score = gs_tree.best_score_

gs_adaBoost = GridSearchCV(AdaBoostRegressor(random_state=42), ab_grid, scoring='r2')
gs_adaBoost.fit(X_train, y_train)
best_ab_score = gs_adaBoost.best_score_

print('Лучшие параметры для DecisionTreeRegressor:', gs_tree.best_params_)
print('Кросс-валидация R2 для DecisionTreeRegressor после подбора', best_tree_score)

print(f"Лучшие параметры для AdaBoostRegressor:", gs_adaBoost.best_params_)
print(f"Кросс-валидация R2 для AdaBoostRegressor после подбора:", best_ab_score)


from time import process_time 

model_tree = DecisionTreeRegressor(max_depth=5, max_features='sqrt', min_samples_split=7, splitter='best', random_state=42)
model_ab = AdaBoostRegressor(learning_rate=0.09, loss='square', n_estimators=90, random_state=42)

# обучим каждую модель несколько раз и построи графики
tree_time = []
for i in range(10) :
    start = process_time()
    tree_fit = model_tree.fit(X_train, y_train)
    end = process_time()
    tree_time.append(end - start)

adab_time = []
for i in range(10) :
    start = process_time()
    ad_fit = model_ab.fit(X_train, y_train)
    end = process_time()
    adab_time.append(end - start)

print(tree_time)
print(adab_time)

plt.figure(figsize=(10,6))
plt.boxplot([tree_time, adab_time], labels=['Decision Tree', 'AdaBoost'])
plt.title('Время обучения алгоритмов')
plt.ylabel('Время (секунды)')
plt.grid(True)
plt.show()


from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)

exp_var = pca.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

plt.figure(figsize=(10, 6))
plt.plot(cum_var, marker='o')
plt.title('Объясненная дисперсия PCA')
plt.xlabel('Количество компонент')
plt.ylabel('Кумулятивная объясненная дисперсия')
plt.grid()
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axvline(x=5, color='g', linestyle='--')  
plt.show()

n_components = 10
pca_optimal = PCA(n_components=n_components)
X_pca = pca_optimal.fit_transform(X)


from time import process_time 
from sklearn.metrics import r2_score

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, train_size=0.7, random_state=42)

tree_grid = {
    'splitter' : ['best', 'random'],
    'max_depth' : [None, 5,6,7,8,10,11,12,13,14,15,16,17,18,20],
    'min_samples_split' : [2,3,4,5,6,7,8,9,10],
    'max_features' : ['auto', 'sqrt', 'log2']
}

adab_grid = {
    'n_estimators' : [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200],
    'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 1],
    'loss' : ['linear', 'square', 'exponential']
}

tree_search_pca = GridSearchCV(DecisionTreeRegressor(random_state=42), tree_grid, cv=5, scoring='r2')
adab_search_pca = GridSearchCV(AdaBoostRegressor(random_state=42), adab_grid, cv=5, scoring='r2')


start_time = process_time()
tree_search_pca.fit(X_train_pca, y_train_pca)
tree_time_pca = process_time() - start_time

start_time = process_time()
adab_search_pca.fit(X_train_pca, y_train_pca)
adab_time_pca = process_time() - start_time


print(f"для Decision Tree R^2: {tree_search_pca.best_score_:.4f}")
print(f"Время обучения Decision Tree: {tree_time_pca:.4f} секунд")
print()
print(f"для AdaBoost R^2: {adab_search_pca.best_score_:.4f}")
print(f"Время обучения AdaBoost: {adab_time_pca:.4f} секунд")


print('ДО PCA')
print(f"Decision Tree R^2: {best_tree_score:.4f}")
print(f"для AdaBoost R^2: {best_ab_score:.4f}")

print('===================================================')

print('ПОСЛЕ PCA')
print(f"Decision Tree R^2: {tree_search_pca.best_score_:.4f}")
print(f"для AdaBoost R^2: {adab_search_pca.best_score_:.4f}")


from sklearn.metrics import mean_absolute_percentage_error

print('Decision Tree c PCA', mean_absolute_percentage_error(tree_search_pca.predict(X_test_pca), y_test_pca))
print('Decision Tree без PCA', mean_absolute_percentage_error(gs_tree.predict(X_test), y_test))


print('adaBoost с PCA', mean_absolute_percentage_error(adab_search_pca.predict(X_test_pca), y_test_pca))
print('adaBoost без PCA', mean_absolute_percentage_error(gs_adaBoost.predict(X_test), y_test))


# Decision Tree c PCA 7.6132699467501315
# Decision Tree без PCA 5.551893112563192
# adaBoost с PCA 9.109739246003716
# adaBoost без PCA 20.858908319650606