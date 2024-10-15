import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import accuracy_score

data = pd.read_csv('eth_cleaned.csv', encoding='utf-8')
data = shuffle(data)
data['flag'] = data['flag'].astype(str)

exchange_data = data[data['flag'] == 'exchange'].sample(n=11180, random_state=42)
non_exchange_data = data[data['flag'] != 'exchange']
data = pd.concat([exchange_data, non_exchange_data])

data.fillna(0, inplace=True)

label_encoder = LabelEncoder()
data['flag'] = label_encoder.fit_transform(data['flag'])

X = data.drop(columns=['address', 'flag'])
y = data['flag']

scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)

train_pool_scaled = Pool(X_train_scaled, y_train)
test_pool_scaled = Pool(X_test_scaled, y_test)

model_scaled = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, early_stopping_rounds=50, loss_function='MultiClass', verbose=100)

for _ in tqdm(range(1)):
    model_scaled.fit(train_pool_scaled, eval_set=test_pool_scaled)

y_train_pred_scaled = model_scaled.predict(X_train_scaled)
y_test_pred_scaled = model_scaled.predict(X_test_scaled)

train_accuracy_scaled = accuracy_score(y_train, y_train_pred_scaled)
test_accuracy_scaled = accuracy_score(y_test, y_test_pred_scaled)

print(f"Train Accuracy (scaled): {train_accuracy_scaled}")
print(f"Test Accuracy (scaled): {test_accuracy_scaled}")

eval_results_scaled = model_scaled.get_evals_result()

train_score_scaled = eval_results_scaled['learn']['MultiClass']
test_score_scaled = eval_results_scaled['validation']['MultiClass']

plt.figure(figsize=(10, 6))
sns.set(style="darkgrid")
plt.plot(train_score_scaled, label='Train Loss (scaled)', color='blue')
plt.plot(test_score_scaled, label='Test Loss (scaled)', color='red')
plt.title('Training and Test Loss (Scaled Data)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=['Train Accuracy', 'Test Accuracy'], y=[train_accuracy_scaled, test_accuracy_scaled])
plt.title('Train and Test Accuracy (Scaled Data)')
plt.show()
