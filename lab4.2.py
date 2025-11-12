import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import warnings

# Ігноруємо попередження про збіжність (ConvergenceWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- 1. Завантаження даних (РУЧНИЙ СПОСІБ) ---

print("Завантаження датасету MNIST (з локального файлу 'mnist.npz')...")

try:
    # Завантажуємо файл 'mnist.npz'
    with np.load('mnist.npz') as data:
        X_train_full = data['x_train']
        y_train_full = data['y_train']
        X_test_full = data['x_test']
        y_test_full = data['y_test']
except FileNotFoundError:
    print("ПОМИЛКА: Файл 'mnist.npz' не знайдено!")
    print(f"Покладіть завантажений 'mnist.npz' у папку:")
    print("/Users/yanayalovska/PycharmProjects/iod_lab1/")
    exit()

print("Дані завантажено. Обробка...")

# "Випрямляємо" дані
X_train_full = X_train_full.reshape(X_train_full.shape[0], 784)
X_test_full = X_test_full.reshape(X_test_full.shape[0], 784)

# Перетворюємо мітки y на рядки
y_train_full = y_train_full.astype(str)
y_test_full = y_test_full.astype(str)

# --- 2. Коректне Зменшення Вибірки ---

TRAIN_SAMPLES = 6000
TEST_SAMPLES = 1000

_, X_train, _, y_train = train_test_split(
    X_train_full, y_train_full,
    test_size=TRAIN_SAMPLES,
    random_state=42,
    stratify=y_train_full
)

_, X_test, _, y_test = train_test_split(
    X_test_full, y_test_full,
    test_size=TEST_SAMPLES,
    random_state=42,
    stratify=y_test_full
)

print(f"Розмір навчальної вибірки (зменшеної): {X_train.shape[0]} спостережень")
print(f"Розмір тестової вибірки (зменшеної): {X_test.shape[0]} спостережень")
print(f"Кількість ознак (пікселів): {X_train.shape[1]}")

# --- 3. Масштабування даних ---
print("Масштабування даних...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Дані готові до навчання.")
print("-" * 40)

# --- 4. Функція для навчання та оцінки ---

def evaluate_model(model, x_tr, y_tr, x_te, y_te, model_name):
    """Навчає модель та виводить метрики."""
    print(f"\n>>> Початок навчання: {model_name} <<<")

    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_te)

    print(f"\n>>> Звіт для моделі: {model_name} <<<\n")

    cm = confusion_matrix(y_te, y_pred)
    print("Матриця плутанини:")
    print(cm)

    report = classification_report(
        y_te,
        y_pred,
        digits=3
    )
    print("\nКласифікаційний звіт:")
    print(report)
    print("-" * 40)

# --- 5. Навчання та Оцінка Моделей ---

# --- А. Логістична Регресія ---

# A1: Логістична регресія (One-vs-Rest / OvR)
lr_base_ovr = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
lr_ovr = OneVsRestClassifier(lr_base_ovr, n_jobs=-1)
evaluate_model(lr_ovr, X_train, y_train, X_test, y_test, "Логістична Регресія (OvR)")

# A2: Логістична регресія (One-vs-One / OvO)
lr_base_ovo = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    random_state=42
    # *** ВИПРАВЛЕНО: Рядок dual='auto' видалено ***
)
lr_ovo = OneVsOneClassifier(lr_base_ovo, n_jobs=-1)
evaluate_model(lr_ovo, X_train, y_train, X_test, y_test, "Логістична Регресія (OvO)")


# --- Б. Метод Опорних Векторів (SVM) ---

# Б1: SVM (One-vs-Rest / OvR)
svm_base_ovr = LinearSVC(
    random_state=42,
    max_iter=2000,
    dual=False
)
svm_ovr = OneVsRestClassifier(svm_base_ovr, n_jobs=-1)
evaluate_model(svm_ovr, X_train, y_train, X_test, y_test, "SVM (OvR) - (LinearSVC)")

# Б2: SVM (One-vs-One / OvO)
svm_base_ovo = LinearSVC(
    random_state=42,
    max_iter=2000,
    dual=False
)
svm_ovo = OneVsOneClassifier(svm_base_ovo, n_jobs=-1)
evaluate_model(svm_ovo, X_train, y_train, X_test, y_test, "SVM (OvO) - (LinearSVC)")
