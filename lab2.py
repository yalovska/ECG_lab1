import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, adjusted_rand_score
import warnings

# --- Допоміжні функції ---

def load_ecg_data(file_path):
    """Завантаження даних ЕКГ"""
    try:
        data = pd.read_csv(file_path, header=None, delimiter=',').values
        return data
    except FileNotFoundError:
        print(f"ПОМИЛКА: Файл ЕКГ не знайдено за шляхом: {file_path}")
        return None
    except Exception as e:
        print(f"ПОМИЛКА при читанні файлу ЕКГ: {e}")
        return None


def load_bike_data(file_path):
    """Завантаження даних про велосипеди (day.csv)"""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"ПОМИЛКА: Файл 'day.csv' не знайдено за шляхом: {file_path}")
        return None
    except Exception as e:
        print(f"ПОМИЛКА при читанні файлу 'day.csv': {e}")
        return None


# =============================================================================
# ОСНОВНА ПРОГРАМА
# =============================================================================

def main():
    # Пригнічуємо зайві попередження для чистішого виводу
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')
    warnings.filterwarnings('ignore', category=FutureWarning)

    # --- Налаштування шляхів до файлів ---
    ecg_file_path = "A18.txt"      # Файл з даними ЕКГ
    bike_day_file_path = "day.csv" # Файл з даними про велосипеди

    # --- Завантаження даних ---
    ecg_data = load_ecg_data(ecg_file_path)
    bike_data = load_bike_data(bike_day_file_path)

    # =========================================================================
    # ЗАВДАННЯ 1: АНАЛІЗ ЕКГ
    # =========================================================================

    if ecg_data is not None:
        print("\n" + "=" * 70)
        print("ЗАВДАННЯ 1: АНАЛІЗ ЕКГ")
        print("1.1. Факторний аналіз")
        print("=" * 70)

        # Нормалізація даних ЕКГ (Z-score)
        scaler_ecg = StandardScaler()
        normalized_data = scaler_ecg.fit_transform(ecg_data)

        # --- Крок 1: Власні числа кореляційної матриці ---
        print("\nКрок 1: Знаходження власних чисел кореляційної матриці")
        n_samples_ecg = normalized_data.shape[0]

        # Кореляційна матриця (r)
        corr_matrix = np.corrcoef(normalized_data.T)

        # Власні числа (lambda) та власні вектори (a_k)
        eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

        # Сортування компонент за спаданням власних чисел
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]

        # Розрахунок поясненої та кумулятивної дисперсії
        total_variance = np.sum(eigenvalues)
        explained_variance = eigenvalues / total_variance
        cumulative_variance = np.cumsum(explained_variance)

        print("Власні числа, частка дисперсії та сумарна дисперсія:")
        variance_data_for_df = {
            'Власні числа': eigenvalues,
            'Частка дисперсії (%)': explained_variance * 100,
            'Сумарна дисперсія (%)': cumulative_variance * 100
        }
        variance_df = pd.DataFrame(variance_data_for_df, index=np.arange(1, 13))
        print(variance_df.round(4))

        # Визначення кількості компонент для 97% дисперсії
        n_components_97 = np.where(cumulative_variance >= 0.97)[0][0] + 1
        print(f"\n{n_components_97} компоненти пояснюють >= 97% дисперсії")

        # --- Крок 2: Критерій "кам'янистого осипу" (Scree Plot) ---
        print("\nКрок 2: Критерій 'кам\'янистого осипу'")
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, 'o-')
        plt.title('Критерій "Кам\'янистий осип" (Scree Plot)')
        plt.xlabel('Номер головної компоненти')
        plt.ylabel('Власне число (Eigenvalue)')
        plt.xticks(np.arange(1, len(eigenvalues) + 1))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        # --- Крок 3: Власні вектори (Матриця L) ---
        # Обираємо 3 головні компоненти (згідно з Scree Plot / умовою)
        k_components = 3
        print(f"\nКрок 3: Матриця власних векторів (перші {k_components} векторів)")
        vector_df = pd.DataFrame(eigenvectors[:, :k_components],
                                 columns=[f'a_{k + 1}' for k in range(k_components)],
                                 index=[f'Канал {i}' for i in range(12)])
        print(vector_df.round(4))

        # --- Крок 4: Перевірка ортонормованості ---
        print("\nКрок 4: Перевірка ортонормованості (a_j' * a_k)")
        # Перевірка: a_j' * a_k == I (одинична матриця)
        ortho_check = np.dot(eigenvectors[:, :k_components].T, eigenvectors[:, :k_components])
        print(ortho_check.round(4))
        print("✓ Діагональні елементи ~1, недіагональні ~0. Умови виконано.")

        # --- Крок 5: Обчислення головних компонент (z_k) ---
        print(f"\nКрок 5: Знаходження головних компонент (z_k = x_norm * a_k)")
        # Використовуємо 'normalized_data', оскільки PCA базується на кореляційній матриці 'r',
        # що вимагає стандартизованих (z-score) даних.
        principal_components_data = np.dot(normalized_data, eigenvectors[:, :k_components])
        pc_df = pd.DataFrame(principal_components_data, columns=[f'Z{k + 1}' for k in range(k_components)])

        print("Перші 8 рядків головних компонент:")
        print(pc_df.head(8).round(6))

        print("\nПобудова графіку першої головної компоненти (Z1)")
        plt.figure(figsize=(12, 6))
        plt.plot(pc_df['Z1'], 'r-', linewidth=1)
        plt.title('Графік першої головної компоненти (Z1)')
        plt.xlabel('Час (i)')
        plt.ylabel('Амплітуда (S1_i)')
        plt.grid(True, alpha=0.3)
        plt.show()

        # --- Крок 6: Перевірка властивостей головних компонент ---
        print("\nКрок 6: Перевірка властивостей головних компонент")
        # 1. Середнє z_k має бути ~0
        means = np.mean(principal_components_data, axis=0)
        print(f"1. Середні значення z_k (мають бути ~0): {means.round(6)}")

        # 2. Дисперсія z_k має дорівнювати lambda_k
        variances = np.var(principal_components_data, axis=0)
        print(f"2. Дисперсії z_k (Var): {variances.round(4)}")
        print(f"   Власні числа l_k: {eigenvalues[:k_components].round(4)}")
        print("   ✓ Дисперсії компонент ~ рівні власним числам.")

        # 3. Коваріація між z_k має бути ~0 (некорельованість)
        cov_z = np.cov(principal_components_data.T)
        print("3. Коваріаційна матриця z_k (недіагональні елементи мають бути ~0):")
        print(cov_z.round(4))
        print("   ✓ Компоненти не корелюють.")

        # --- Крок 7: Висновки факторного аналізу ЕКГ ---
        print("\nКрок 7: Висновки факторного аналізу")
        print(f"✓ Факторний аналіз успішно проведено.")
        print(
            f"✓ Розмірність даних знижено з 12 до {k_components}, зберігши {cumulative_variance[k_components - 1] * 100:.2f}% дисперсії.")

        # =========================================================================
        # 1.2. Кластерний аналіз
        # =========================================================================
        print("\n" + "=" * 70)
        print("1.2. Кластерний аналіз (k-Means)")
        print("=" * 70)

        # --- А) Кластеризація 12D (початкових) даних ---
        print("\nА) Кластеризація 12-вимірних нормалізованих даних")

        # k=11
        kmeans_11_12D = KMeans(n_clusters=11, random_state=42, n_init=10)
        clusters_11_12D = kmeans_11_12D.fit_predict(normalized_data)
        centers_11_12D = kmeans_11_12D.cluster_centers_
        print(f"✓ Дані (12D) кластеризовано на k=11 кластерів")

        # k=7
        kmeans_7_12D = KMeans(n_clusters=7, random_state=42, n_init=10)
        clusters_7_12D = kmeans_7_12D.fit_predict(normalized_data)
        centers_7_12D = kmeans_7_12D.cluster_centers_
        print(f"✓ Дані (12D) кластеризовано на k=7 кластерів")

        # --- Візуалізація кластерів (A) (проекція на PC1/PC2) ---
        pc_vis = principal_components_data[:, :2]  # Z1 (Feature 1) and Z2 (Feature 2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Графік для k=11
        ax1.scatter(pc_vis[:, 0], pc_vis[:, 1], c=clusters_11_12D, cmap='tab20', s=5, alpha=0.7)
        centers_11_proj = np.dot(centers_11_12D, eigenvectors[:, :2])
        ax1.scatter(centers_11_proj[:, 0], centers_11_proj[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
        ax1.set_title('A) 12D-дані, кластеризовані на k=11 (Проекція на PC1, PC2)')
        ax1.set_xlabel('Feature 1 (PC1)')
        ax1.set_ylabel('Feature 2 (PC2)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.4)

        # Графік для k=7
        ax2.scatter(pc_vis[:, 0], pc_vis[:, 1], c=clusters_7_12D, cmap='tab10', s=5, alpha=0.7)
        centers_7_proj = np.dot(centers_7_12D, eigenvectors[:, :2])
        ax2.scatter(centers_7_proj[:, 0], centers_7_proj[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
        ax2.set_title('A) 12D-дані, кластеризовані на k=7 (Проекція на PC1, PC2)')
        ax2.set_xlabel('Feature 1 (PC1)')
        ax2.set_ylabel('Feature 2 (PC2)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.4)
        plt.show()

        # --- Б) Кластеризація 3D (головних компонент) даних ---
        print(f"\nБ) Кластеризація 3-вимірних даних (z_k)")
        # k=11
        kmeans_11_3D = KMeans(n_clusters=11, random_state=42, n_init=10)
        clusters_11_3D = kmeans_11_3D.fit_predict(principal_components_data)
        centers_11_3D = kmeans_11_3D.cluster_centers_
        print(f"✓ 3D дані (PC) кластеризовано на k=11 кластерів")

        # k=7
        kmeans_7_3D = KMeans(n_clusters=7, random_state=42, n_init=10)
        clusters_7_3D = kmeans_7_3D.fit_predict(principal_components_data)
        centers_7_3D = kmeans_7_3D.cluster_centers_
        print(f"✓ 3D дані (PC) кластеризовано на k=7 кластерів")

        # --- Візуалізація кластерів (Б) ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Графік для k=11 (3D)
        ax1.scatter(pc_vis[:, 0], pc_vis[:, 1], c=clusters_11_3D, cmap='tab20', s=5, alpha=0.7)
        ax1.scatter(centers_11_3D[:, 0], centers_11_3D[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
        ax1.set_title('Б) 3D-дані (PC), кластеризовані на k=11')
        ax1.set_xlabel('Feature 1 (PC1)')
        ax1.set_ylabel('Feature 2 (PC2)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.4)

        # Графік для k=7 (3D)
        ax2.scatter(pc_vis[:, 0], pc_vis[:, 1], c=clusters_7_3D, cmap='tab10', s=5, alpha=0.7)
        ax2.scatter(centers_7_3D[:, 0], centers_7_3D[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
        ax2.set_title('Б) 3D-дані (PC), кластеризовані на k=7')
        ax2.set_xlabel('Feature 1 (PC1)')
        ax2.set_ylabel('Feature 2 (PC2)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.4)
        plt.show()

        # --- Порівняння кластеризацій (A vs Б) через ARI ---
        print("\nПорівняння кластерів (A: 12D vs B: 3D)")

        ari_11 = adjusted_rand_score(clusters_11_12D, clusters_11_3D)
        ari_7 = adjusted_rand_score(clusters_7_12D, clusters_7_3D)

        print(f"Adjusted Rand Index (ARI) для k=11: {ari_11:.4f}")
        print(f"Adjusted Rand Index (ARI) для k=7:  {ari_7:.4f}")
        print("(ARI = 1.0 означає ідеальний збіг, 0.0 - випадковий)")
        if ari_11 > 0.8 and ari_7 > 0.8:
            print("✓ Кластери у варіантах А) і Б) дуже схожі для обох k.")
        else:
            print("✗ Кластери у варіантах А) і Б) суттєво відрізняються.")

        # --- Визначення R-піків на основі кластеризації ---
        print("\nВизначення R-піків (на основі кластеризації A, k=11)")

        pc_df['Cluster_12D_k11'] = clusters_11_12D

        # R-піки - це екстремальні значення, шукаємо кластер з max(mean(Z1))
        cluster_means_z1 = pc_df.groupby('Cluster_12D_k11')['Z1'].mean()
        r_peak_cluster_label = cluster_means_z1.idxmax()

        print(f"Середні 'Z1' для кластерів (k=11):\n{cluster_means_z1.sort_values(ascending=False)}")
        print(f"Припускаємо, що R-піки - це кластер {r_peak_cluster_label}")

        r_peak_indices = pc_df[pc_df['Cluster_12D_k11'] == r_peak_cluster_label].index

        # Формуємо сигнал R-піків для візуалізації
        r_peak_signal = np.zeros(n_samples_ecg)
        # Ставимо високе значення у точках піків
        r_peak_signal[r_peak_indices] = 5000

        plt.figure(figsize=(12, 6))
        plt.plot(r_peak_signal, 'b-', linewidth=1.5)
        plt.title('Визначені R-піки (з кластеризації k=11)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.ylim(bottom=0, top=max(5500, np.max(r_peak_signal) * 1.1))
        plt.grid(True, alpha=0.3)
        plt.show()

    else:
        print("\nПропускаємо Завдання 1 (Аналіз ЕКГ), оскільки дані не завантажено.")

    # =========================================================================
    # ЗАВДАННЯ 2: АНАЛІЗ ОРЕНДИ ВЕЛОСИПЕДІВ
    # =========================================================================

    if bike_data is not None:
        print("\n" + "=" * 70)
        print("ЗАВДАННЯ 2: АНАЛІЗ ОРЕНДИ ВЕЛОСИПЕДІВ")
        print("Використовуємо 'day.csv'")
        print("=" * 70)

        # --- 2.1. Факторний аналіз (Bike Data) ---
        print("\n2.1. Факторний аналіз (Bike Data)")

        # Відображення назв (PDF -> CSV):
        # 'температура1' -> 'temp'
        # 'температура2' -> 'atemp'
        # 'шум' -> 'hum' (вологість)
        # 'швидкість вітру' -> 'windspeed'
        # 'підсумкова кількість пасажирів' -> 'cnt'
        factor_cols = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']

        if not all(col in bike_data.columns for col in factor_cols):
            print(f"ПОМИЛКА: В 'day.csv' відсутні необхідні колонки: {factor_cols}")
        else:
            bike_factor_data = bike_data[factor_cols]

            # Нормалізація даних
            scaler_bike = StandardScaler()
            bike_factor_data_norm = scaler_bike.fit_transform(bike_factor_data)

            # Кореляційна матриця
            corr_matrix_bike = np.corrcoef(bike_factor_data_norm.T)
            print("Кореляційна матриця:")
            corr_df_bike = pd.DataFrame(corr_matrix_bike, columns=factor_cols, index=factor_cols)
            print(corr_df_bike.round(4))

            # Власні числа та вектори
            eigenvalues_bike, eigenvectors_bike = np.linalg.eig(corr_matrix_bike)

            sort_indices_bike = np.argsort(eigenvalues_bike)[::-1]
            eigenvalues_bike = eigenvalues_bike[sort_indices_bike]
            eigenvectors_bike = eigenvectors_bike[:, sort_indices_bike]

            # Виділення 2 головних компонент (за умовою)
            n_components_bike = 2
            pc_bike_data = np.dot(bike_factor_data_norm, eigenvectors_bike[:, :n_components_bike])

            print(f"\n{n_components_bike} головні компоненти (Z1, Z2):")
            pc_bike_df = pd.DataFrame(pc_bike_data, columns=['Z1_bike', 'Z2_bike'])
            print(pc_bike_df.head())

            # Розрахунок поясненої дисперсії для 2 компонент
            total_variance_bike = np.sum(eigenvalues_bike)
            explained_variance_bike = np.sum(eigenvalues_bike[:n_components_bike]) / total_variance_bike
            print(f"\n{n_components_bike} компоненти пояснюють {explained_variance_bike * 100:.2f}% дисперсії.")

        # --- 2.2. Регресійний аналіз ---
        print("\n2.2. Регресійний аналіз")

        # Цільові (залежні) змінні
        target_vars = ['casual', 'registered']

        # Набори незалежних змінних (предиктори)
        feature_sets = {
            'Model_temp (проста)': ['temp'],
            'Model_wind (проста)': ['windspeed'],
            'Model_multi (декілька)': ['temp', 'atemp', 'hum', 'windspeed']
        }

        regression_results = []

        for target in target_vars:
            print(f"\n--- Моделі для залежної змінної: '{target}' ---")

            if target not in bike_data.columns:
                print(f"ПОМИЛКА: Цільова колонка '{target}' відсутня.")
                continue

            y = bike_data[target]

            for model_name, features in feature_sets.items():
                if not all(col in bike_data.columns for col in features):
                    print(f"ПОМИЛКА: Пропуск {model_name}, відсутні колонки: {features}")
                    continue

                X = bike_data[features]

                # Навчання моделі
                model = LinearRegression()
                model.fit(X, y)

                y_pred = model.predict(X)

                # Оцінка
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)

                print(f"  {model_name} (Features: {features}):")
                print(f"    Рівняння: {target} = {model.intercept_:.4f} + " + \
                      " + ".join([f"({coef:.4f} * {name})" for coef, name in zip(model.coef_, features)]))
                print(f"    R^2: {r2:.4f}, MSE: {mse:.4f}")

                regression_results.append({
                    'Target': target,
                    'Model': model_name,
                    'Features': ", ".join(features),
                    'R2': r2,
                    'MSE': mse
                })

                # Графік для простих моделей (з однією змінною)
                if len(features) == 1:
                    plt.figure(figsize=(8, 5))
                    plt.scatter(X[features[0]], y, alpha=0.3, label='Фактичні дані')
                    plt.plot(X[features[0]], y_pred, color='red', linewidth=2, label='Лінія регресії')
                    plt.title(f"Регресія: {target} ~ {features[0]}")
                    plt.xlabel(features[0])
                    plt.ylabel(target)
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.show()

        print("\nПідсумок регресійного аналізу:")
        results_df = pd.DataFrame(regression_results)
        print(results_df.round(4))

    else:
        print("\nПропускаємо Завдання 2 (Аналіз велосипедів), оскільки дані не завантажено.")

    # --- Завершення роботи ---
    print("\n" + "=" * 70)
    print("РОБОТУ СКРИПТА ЗАВЕРШЕНО.")
    print("=" * 70)


if __name__ == "__main__":
    main()
