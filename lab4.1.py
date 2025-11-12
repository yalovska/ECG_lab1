import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. Підготовка даних ---

# Завантажуємо датасет Ірисів
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Розділяємо вибірку: 70% на навчання, 30% на тестування
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Масштабування даних (рекомендовано для SVM та Лог. Регресії)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Розмір навчальної вибірки: {X_train.shape[0]} спостережень")
print(f"Розмір тестової вибірки: {X_test.shape[0]} спостережень")
print("-" * 40)

# --- 2. Функція для навчання та оцінки ---

def evaluate_model(model, x_tr, y_tr, x_te, y_te, model_name):
    """Навчає модель та виводить метрики."""
    
    # Навчання моделі
    model.fit(x_tr, y_tr)
    
    # Отримання прогнозів
    y_pred = model.predict(x_te)
    
    print(f"\n>>> Звіт для моделі: {model_name} <<<\n")
    
    # Обчислення матриці плутанини
    cm = confusion_matrix(y_te, y_pred)
    print("Матриця плутанини:")
    print(cm)
    
    # Обчислення precision, recall, f1-score, accuracy
    report = classification_report(
        y_te, 
        y_pred, 
        target_names=target_names,
        digits=3
    )
    print("\nКласифікаційний звіт:")
    print(report)
    print("-" * 40)

# --- 3. Навчання та Оцінка Моделей ---

# --- А. Логістична Регресія ---

# A1: Логістична регресія з підходом One-vs-Rest (OvR)
# (Використовуємо обгортку OneVsRestClassifier для коректності)
lr_base_ovr = LogisticRegression(solver='liblinear', random_state=42)
lr_ovr = OneVsRestClassifier(lr_base_ovr)
evaluate_model(lr_ovr, X_train, y_train, X_test, y_test, "Логістична Регресія (OvR)")

# A2: Логістична регресія з підходом One-vs-One (OvO)
# (Використовуємо обгортку OneVsOneClassifier)
lr_base_ovo = LogisticRegression(solver='liblinear', random_state=42)
lr_ovo = OneVsOneClassifier(lr_base_ovo)
evaluate_model(lr_ovo, X_train, y_train, X_test, y_test, "Логістична Регресія (OvO)")


# --- Б. Метод Опорних Векторів (SVM) ---

# Б1: SVM з підходом One-vs-Rest (OvR)
# (Використовуємо обгортку OneVsRestClassifier)
svm_base_ovr = SVC(kernel='linear', probability=True, random_state=42)
svm_ovr = OneVsRestClassifier(svm_base_ovr)
evaluate_model(svm_ovr, X_train, y_train, X_test, y_test, "SVM (OvR)")

# Б2: SVM з підходом One-vs-One (OvO)
# (SVC за замовчуванням використовує OvO)
svm_ovo = SVC(kernel='linear', decision_function_shape='ovo', random_state=42)
evaluate_model(svm_ovo, X_train, y_train, X_test, y_test, "SVM (OvO) - (За замовчуванням)")
