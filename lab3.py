import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -------------------------------------------------------------------
# ЗАВДАННЯ 1: Кластеризація (Сферичні кластери)
# -------------------------------------------------------------------
def run_task_1():
    print("--- Запуск Завдання 1 ---")
    
    # 1. Моделювання даних (5 сферичних кластерів)
    N = 100
    means = [
        [0, 0, 0], [1, 1, 0], [0, 1, 1], [-1, 0, -1], [0, -1, 0]
    ]
    # Одинична матриця для сферичних кластерів
    cov_task1 = np.identity(3)
    
    X_list = []
    y_list = []
    for i in range(5):
        data = np.random.multivariate_normal(means[i], cov_task1, N)
        X_list.append(data)
        y_list.append(np.full(N, i))
    
    X = np.concatenate(X_list)
    y_true = np.concatenate(y_list)

    # 2. Візуалізація початкових даних
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_true, cmap='viridis', alpha=0.7)
    ax.set_title('Завдання 1: Початкові 5 кластерів (Сферичні)')
    plt.show()

    # 3. Кластеризація K-Means (k=3, 4, 5)
    wcss_scores = {}
    for k in [3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        wcss_scores[f'K={k}'] = kmeans.inertia_
        
        print(f"\nЗавдання 1: K-Means (k={k})")
        print(f"  WCSS: {kmeans.inertia_:.4f}")
        print("  Центри кластерів:")
        print(pd.DataFrame(kmeans.cluster_centers_))

        # Візуалізація K-Means
        fig_k = plt.figure(figsize=(10, 8))
        ax_k = fig_k.add_subplot(111, projection='3d')
        ax_k.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans.labels_, cmap='viridis', alpha=0.5)
        ax_k.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], 
                     c='red', marker='X', s=200)
        ax_k.set_title(f'Завдання 1: K-Means (k={k})')
        plt.show()

    # 4. Кластеризація DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=6)
    labels_dbscan = dbscan.fit_predict(X)
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    print(f"\nЗавдання 1: DBSCAN знайшов {n_clusters_dbscan} кластерів.")
    
    # Розрахунок WCSS для DBSCAN
    wcss_dbscan = 0
    centers_dbscan = []
    for label in set(labels_dbscan):
        if label != -1:
            cluster_points = X[labels_dbscan == label]
            center = np.mean(cluster_points, axis=0)
            centers_dbscan.append(center)
            wcss_dbscan += np.sum((cluster_points - center) ** 2)
            
    wcss_scores['DBSCAN'] = wcss_dbscan
    print(f"  WCSS (DBSCAN): {wcss_dbscan:.4f}")

    # Візуалізація DBSCAN
    fig_db = plt.figure(figsize=(10, 8))
    ax_db = fig_db.add_subplot(111, projection='3d')
    ax_db.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels_dbscan, cmap='viridis', alpha=0.5)
    if centers_dbscan:
        ax_db.scatter(np.array(centers_dbscan)[:, 0], np.array(centers_dbscan)[:, 1], np.array(centers_dbscan)[:, 2], 
                      c='red', marker='X', s=200)
    ax_db.set_title(f'Завдання 1: DBSCAN (знайдено {n_clusters_dbscan} кластерів)')
    plt.show()

    print("\nЗавдання 1: Загальна якість (WCSS)")
    print(wcss_scores)


# -------------------------------------------------------------------
# ЗАВДАННЯ 2: Кластеризація (Еліптичні кластери)
# -------------------------------------------------------------------
def run_task_2():
    print("\n\n--- Запуск Завдання 2 ---")
    
    # 1. Моделювання даних (5 еліптичних кластерів)
    N = 100
    means = [
        [0, 0, 0], [1, 1, 0], [0, 1, 1], [-1, 0, -1], [0, -1, 0]
    ]
    # Матриця коваріації 3x3 з умови
    cov_task2 = np.array([
        [1.0, 0.8, 0.6],
        [0.8, 1.0, 0.9],
        [0.6, 0.9, 1.0]
    ])
    
    X_list = []
    y_list = []
    for i in range(5):
        data = np.random.multivariate_normal(means[i], cov_task2, N)
        X_list.append(data)
        y_list.append(np.full(N, i))
    
    X = np.concatenate(X_list)
    y_true = np.concatenate(y_list)

    # 2. Візуалізація початкових даних
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_true, cmap='viridis', alpha=0.7)
    ax.set_title('Завдання 2: Початкові 5 кластерів (Еліптичні)')
    plt.show()

    # 3. Кластеризація K-Means (k=3, 4, 5)
    wcss_scores = {}
    for k in [3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        wcss_scores[f'K={k}'] = kmeans.inertia_
        
        print(f"\nЗавдання 2: K-Means (k={k})")
        print(f"  WCSS: {kmeans.inertia_:.4f}")
        print("  Центри кластерів:")
        print(pd.DataFrame(kmeans.cluster_centers_))

        # Візуалізація K-Means
        fig_k = plt.figure(figsize=(10, 8))
        ax_k = fig_k.add_subplot(111, projection='3d')
        ax_k.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans.labels_, cmap='viridis', alpha=0.5)
        ax_k.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], 
                     c='red', marker='X', s=200)
        ax_k.set_title(f'Завдання 2: K-Means (k={k})')
        plt.show()

    # 4. Кластеризація DBSCAN
    # Потрібен більший 'eps' через "розтягнуті" кластери
    dbscan = DBSCAN(eps=2.5, min_samples=6) 
    labels_dbscan = dbscan.fit_predict(X)
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    print(f"\nЗавдання 2: DBSCAN знайшов {n_clusters_dbscan} кластерів.")
    
    # Розрахунок WCSS для DBSCAN
    wcss_dbscan = 0
    centers_dbscan = []
    for label in set(labels_dbscan):
        if label != -1:
            cluster_points = X[labels_dbscan == label]
            center = np.mean(cluster_points, axis=0)
            centers_dbscan.append(center)
            wcss_dbscan += np.sum((cluster_points - center) ** 2)
            
    wcss_scores['DBSCAN'] = wcss_dbscan
    print(f"  WCSS (DBSCAN): {wcss_dbscan:.4f}")

    # Візуалізація DBSCAN
    fig_db = plt.figure(figsize=(10, 8))
    ax_db = fig_db.add_subplot(111, projection='3d')
    ax_db.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels_dbscan, cmap='viridis', alpha=0.5)
    if centers_dbscan:
        ax_db.scatter(np.array(centers_dbscan)[:, 0], np.array(centers_dbscan)[:, 1], np.array(centers_dbscan)[:, 2], 
                      c='red', marker='X', s=200)
    ax_db.set_title(f'Завдання 2: DBSCAN (знайдено {n_clusters_dbscan} кластерів)')
    plt.show()

    print("\nЗавдання 2: Загальна якість (WCSS)")
    print(wcss_scores)


# -------------------------------------------------------------------
# ЗАВДАННЯ 3: Логістична регресія
# -------------------------------------------------------------------
def run_task_3():
    print("\n\n--- Запуск Завдання 3 ---")
    
    # 1. Генерація даних (Квадрат [0,1]x[0,1])
    N = 500
    X = np.random.uniform(0, 1, (N, 2))
    # Класи: y > x (Клас 1), y < x (Клас 0)
    y_true = np.where(X[:, 1] > X[:, 0], 1, 0)

    # 2. Навчання моделі логістичної регресії
    model = LogisticRegression(random_state=42)
    model.fit(X, y_true)
    y_pred = model.predict(X)

    # 3. Оцінка якості (Матриця плутанини та метрики)
    cm = confusion_matrix(y_true, y_pred)
    print("Завдання 3: Оцінки точності")
    print("Матриця плутанини:")
    print(cm)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred):.4f}")

    # 4. Візуалізація (Теплова карта та межа)
    plt.figure(figsize=(10, 8))

    # Створення сітки для теплової карти
    xx, yy = np.meshgrid(np.linspace(0, 1, 300), 
                         np.linspace(0, 1, 300))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Теплова карта
    plt.contourf(xx, yy, Z, cmap='RdBu', alpha=0.7, levels=np.linspace(0, 1, 100))
    plt.colorbar(label='P(class=1)')

    # Точки
    plt.scatter(X[y_true == 0, 0], X[y_true == 0, 1], 
                c='blue', label='Клас 0 (y < x)', alpha=0.6, edgecolors='k')
    plt.scatter(X[y_true == 1, 0], X[y_true == 1, 1], 
                c='orange', label='Клас 1 (y > x)', alpha=0.6, edgecolors='k')

    # Межа рішення (модель)
    coef = model.coef_[0]
    intercept = model.intercept_
    x_boundary = np.array([0, 1])
    y_boundary = -(intercept + coef[0] * x_boundary) / coef[1]
    plt.plot(x_boundary, y_boundary, color='green', linewidth=3, 
             label='Межа рішення (модель)')
    
    # Ідеальна межа (y=x)
    plt.plot([0, 1], [0, 1], 'k--', label='Ідеальна межа (y=x)')

    plt.title('Завдання 3: Логістична регресія: Класифікація y > x')
    plt.xlabel('X1 (x)')
    plt.ylabel('X2 (y)')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

    # Візуалізація матриці плутанини
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Пред. 0', 'Пред. 1'], 
                yticklabels=['Справж. 0', 'Справж. 1'])
    plt.title('Завдання 3: Матриця плутанини')
    plt.ylabel('Справжній клас')
    plt.xlabel('Передбачений клас')
    plt.show()


# -------------------------------------------------------------------
# ГОЛОВНИЙ БЛОК ВИКОНАННЯ
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Послідовно запускаємо кожне завдання
    run_task_1()
    run_task_2()
    run_task_3()
    
    print("\n\n--- Всі завдання виконано ---")
