import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, skew, kurtosis, f

def load_data(file_path):
    """Завантаження даних ЕКГ"""
    return pd.read_csv(file_path, header=None, delimiter=',').values


def fast_gini_difference(data):
    """Швидке обчислення середньої різниці Джині"""
    n = len(data)
    sorted_data = np.sort(data)
    total = np.sum(sorted_data)

    if total == 0:
        return np.nan

    gini_sum = np.sum((2 * np.arange(n) - n + 1) * sorted_data)
    return gini_sum / (n * total)


def partial_correlation(r_ab, r_ac, r_bc):
    """Обчислення часткового коефіцієнта кореляції"""
    numerator = r_ab - r_ac * r_bc
    denominator = np.sqrt((1 - r_ac ** 2) * (1 - r_bc ** 2))
    return numerator / denominator if denominator != 0 else 0


def multiple_correlation_2var(r_ab, r_ac, r_bc):
    """Множинний коефіцієнт кореляції для двох факторів"""
    numerator = r_ab ** 2 + r_ac ** 2 - 2 * r_ab * r_ac * r_bc
    denominator = 1 - r_bc ** 2
    return np.sqrt(numerator / denominator) if denominator > 0 else 0


def multiple_correlation_3var(r_ab, r_ac_b, r_ad_bc):
    """Множинний коефіцієнт кореляції для трьох факторів"""
    R_squared = 1 - (1 - r_ab ** 2) * (1 - r_ac_b ** 2) * (1 - r_ad_bc ** 2)
    return np.sqrt(max(0, R_squared))


def fourier_transform(signal):
    """Пряме перетворення Фур'є"""
    N = len(signal)
    j_indices = np.arange(N // 2 + 1)

    # Обчислення коефіцієнтів A та B
    A = np.zeros(N // 2 + 1)
    B = np.zeros(N // 2 + 1)

    for j in j_indices:
        if j == 0:
            A[j] = np.mean(signal)
        elif j == N // 2:
            A[j] = np.mean(signal * np.cos(np.pi * np.arange(N)))
        else:
            A[j] = 2 * np.mean(signal * np.cos(2 * np.pi * j * np.arange(N) / N))

        B[j] = 2 * np.mean(signal * np.sin(2 * np.pi * j * np.arange(N) / N))

    # Амплітудний спектр
    C = np.sqrt(A ** 2 + B ** 2)

    return A, B, C


def inverse_fourier_transform(A, B, N):
    """Обернене перетворення Фур'є"""
    reconstructed = np.zeros(N)

    for i in range(N):
        cos_sum = np.sum(A * np.cos(2 * np.pi * np.arange(len(A)) * i / N))
        sin_sum = np.sum(B * np.sin(2 * np.pi * np.arange(len(B)) * i / N))
        reconstructed[i] = cos_sum + sin_sum

    return reconstructed


# =============================================================================
# ОСНОВНА ПРОГРАМА
# =============================================================================

def main():
    # Завантаження даних
    file_path = "A9/A1.txt"
    ecg_data = load_data(file_path)
    print(f"Розмірність даних: {ecg_data.shape}")

    # =========================================================================
    # 1. ПОПЕРЕДНІЙ АНАЛІЗ ДАНИХ
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. ПОПЕРЕДНІЙ АНАЛІЗ ДАНИХ")
    print("=" * 70)

    # 1.1. Побудова графіків кардіограми
    print("\n1.1. Побудова графіків кардіограми...")
    fig, axes = plt.subplots(6, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(12):
        axes[i].plot(ecg_data[:, i], linewidth=0.8)
        axes[i].set_title(f'Канал {i + 1}')
        axes[i].set_xlabel('Час')
        axes[i].set_ylabel('Амплітуда')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 1.2. Обчислення статистичних параметрів
    print("\n1.2. Обчислення статистичних параметрів...")

    statistics_data = []
    for i in range(12):
        channel_data = ecg_data[:, i]

        # Основні статистики
        mean_val = np.mean(channel_data)
        harmonic_mean_val = len(channel_data) / np.sum(1.0 / (np.abs(channel_data) + 1e-10))

        # Геометричне середнє
        positive_data = channel_data[channel_data > 0]
        geometric_mean_val = np.exp(np.mean(np.log(positive_data))) if len(positive_data) > 0 else np.nan

        # Інші статистики
        variance_val = np.var(channel_data)
        std_val = np.std(channel_data)

        # Мода (спрощений підхід)
        hist, bin_edges = np.histogram(channel_data, bins=50)
        mode_val = bin_edges[np.argmax(hist)]

        median_val = np.median(channel_data)
        skewness_val = skew(channel_data)
        kurtosis_val = kurtosis(channel_data)
        gini_val = fast_gini_difference(channel_data)

        statistics_data.append([
            mean_val, harmonic_mean_val, geometric_mean_val, variance_val, std_val,
            gini_val, mode_val, median_val, skewness_val, kurtosis_val
        ])

    # DataFrame зі статистикою
    statistics_df = pd.DataFrame(
        statistics_data,
        index=[f'Канал {i + 1}' for i in range(12)],
        columns=[
            'Середнє', 'Середнє гармонічне', 'Середнє геометричне',
            'Дисперсія', 'Стандартне відхилення', 'Середня різниця Джині',
            'Мода', 'Медіана', 'Коефіцієнт асиметрії', 'Коефіцієнт ексцесу'
        ]
    )
    print(statistics_df.round(4))

    # 1.3. Побудова гістограм
    print("\n1.3. Побудова гістограм...")
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(12):
        axes[i].hist(ecg_data[:, i], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Канал {i + 1}')
        axes[i].set_xlabel('Амплітуда')
        axes[i].set_ylabel('Частота')

    # Прибираємо зайві субплoти
    for i in range(12, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

    # 1.4. Перевірка нормальності
    print("\n1.4. Перевірка нормальності розподілу...")
    shapiro_results = []

    for i in range(min(6, 12)):
        sample_data = ecg_data[:1000, i] if len(ecg_data) > 1000 else ecg_data[:, i]
        _, p_value = shapiro(sample_data)
        is_normal = "Так" if p_value > 0.05 else "Ні"
        shapiro_results.append([p_value, is_normal])

    shapiro_df = pd.DataFrame(
        shapiro_results,
        index=[f'Канал {i + 1}' for i in range(min(6, 12))],
        columns=['p-value', 'Нормальний розподіл']
    )
    print(shapiro_df.round(6))

    # 1.5. Нормалізація даних
    print("\n1.5. Нормалізація даних...")
    normalized_data = np.zeros_like(ecg_data)
    for i in range(12):
        channel = ecg_data[:, i]
        normalized_data[:, i] = (channel - np.mean(channel)) / np.std(channel)

    print("Перевірка нормалізації:")
    for i in range(3):
        mean_after = np.mean(normalized_data[:, i])
        var_after = np.var(normalized_data[:, i])
        print(f'Канал {i + 1}: μ={mean_after:.6f}, σ²={var_after:.6f}')

    # =========================================================================
    # 2. ОДНОФАКТОРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. ОДНОФАКТОРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ")
    print("=" * 70)

    k, n = 12, ecg_data.shape[0]
    print(f"Кількість рівнів фактора (каналів): k = {k}")
    print(f"Кількість спостережень на рівень: n = {n}")

    # Обчислення дисперсій
    S_i_squared, x_i_bar = [], []
    for i in range(k):
        x_i = ecg_data[:, i]
        x_i_bar.append(np.mean(x_i))
        S_i_squared.append(np.var(x_i, ddof=1))

    # Оцінки дисперсій
    S_0_squared = np.mean(S_i_squared)
    S_A_squared = (n / (k - 1)) * np.var(x_i_bar, ddof=1) * k

    # F-критерій
    F_observed = S_A_squared / S_0_squared
    df1, df2 = k - 1, k * (n - 1)
    F_critical = f.ppf(0.95, df1, df2)
    p_value = 1 - f.cdf(F_observed, df1, df2)

    print(f"F-статистика = {F_observed:.4f}")
    print(f"F-критичне (0.05, {df1}, {df2}) = {F_critical:.4f}")
    print(f"p-value = {p_value:.6f}")

    if F_observed > F_critical:
        print("✓ ВПЛИВ ФАКТОРА ЗНАЧУЩИЙ: F_observed > F_critical")
    else:
        print("✗ ВПЛИВ ФАКТОРА НЕЗНАЧУЩИЙ: F_observed ≤ F_critical")

    # =========================================================================
    # 3. ДВОФАКТОРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. ДВОФАКТОРНИЙ ДИСПЕРСІЙНИЙ АНАЛІЗ")
    print("=" * 70)

    k, m, n_part = 12, 5, 1000
    print(f"Фактор A (канали): k = {k} рівнів")
    print(f"Фактор B (частини): m = {m} рівнів")
    print(f"Спостережень у кожній клітинці: n = {n_part}")

    # Побудова таблиці двофакторного експерименту
    print("\n3.1. Побудова таблиці двофакторного експерименту...")

    two_factor_table = np.zeros((m, k, n_part))
    for i in range(k):
        for j in range(m):
            start_idx = j * n_part
            end_idx = (j + 1) * n_part
            if end_idx <= ecg_data.shape[0]:
                two_factor_table[j, i, :] = ecg_data[start_idx:end_idx, i]

    print("✓ Таблицю побудовано успішно")
    print(f"Розмірність таблиці: {two_factor_table.shape}")

    # Таблиця середніх значень
    print("\n3.2. Таблиця середніх значень двофакторного експерименту:")

    x_ij = np.mean(two_factor_table, axis=2)
    two_factor_df = pd.DataFrame(
        x_ij,
        columns=[f'A{i + 1}' for i in range(k)],
        index=[f'B{j + 1}' for j in range(m)]
    )
    print(two_factor_df.round(4))

    # Теплова карта
    plt.figure(figsize=(14, 6))
    plt.imshow(x_ij, cmap='viridis', aspect='auto')
    plt.colorbar(label='Середнє значення')
    plt.title('Теплова карта середніх значень двофакторного експерименту', fontsize=14)
    plt.xlabel('Фактор A (Канали)')
    plt.ylabel('Фактор B (Частини)')
    plt.xticks(ticks=np.arange(k), labels=[f'A{i + 1}' for i in range(k)])
    plt.yticks(ticks=np.arange(m), labels=[f'B{j + 1}' for j in range(m)])

    for i in range(m):
        for j in range(k):
            plt.text(j, i, f'{x_ij[i, j]:.2f}', ha='center', va='center',
                     color='white' if x_ij[i, j] > np.mean(x_ij) else 'black', fontsize=8)

    plt.tight_layout()
    plt.show()

    # Обчислення основних показників
    print("\n3.3. Обчислення основних показників:")

    Q1 = np.sum(x_ij ** 2)
    Q2 = m * np.sum(np.sum(x_ij, axis=0) ** 2)
    Q3 = k * np.sum(np.sum(x_ij, axis=1) ** 2)
    Q4 = np.sum(x_ij) ** 2 / (m * k)

    print(f"Q1 = ΣΣx_ij² = {Q1:.4f}")
    print(f"Q2 = m * ΣX_i² = {Q2:.4f}")
    print(f"Q3 = k * ΣX_j² = {Q3:.4f}")
    print(f"Q4 = (ΣΣx_ij)²/(m*k) = {Q4:.4f}")

    # Оцінки дисперсій
    S0_squared = (Q1 + Q4 - Q2 - Q3) / ((k - 1) * (m - 1))
    SA_squared = (Q2 - Q4) / (k - 1)
    SB_squared = (Q3 - Q4) / (m - 1)

    print(f"\nS0² (залишкова) = {S0_squared:.4f}")
    print(f"SA² (фактор A) = {SA_squared:.4f}")
    print(f"SB² (фактор B) = {SB_squared:.4f}")

    # Перевірка значущості факторів
    F_A, F_B = SA_squared / S0_squared, SB_squared / S0_squared

    df1_A, df2_A = k - 1, (k - 1) * (m - 1)
    df1_B, df2_B = m - 1, (k - 1) * (m - 1)

    F_crit_A = f.ppf(0.95, df1_A, df2_A)
    F_crit_B = f.ppf(0.95, df1_B, df2_B)

    print(f"\nF_A = {F_A:.4f}, F-критичне = {F_crit_A:.4f}")
    print(f"F_B = {F_B:.4f}, F-критичне = {F_crit_B:.4f}")

    print("✓ ФАКТОР А (КАНАЛИ) ЗНАЧУЩИЙ" if F_A > F_crit_A else "✗ ФАКТОР А (КАНАЛИ) НЕЗНАЧУЩИЙ")
    print("✓ ФАКТОР B (ЧАСТИНИ) ЗНАЧУЩИЙ" if F_B > F_crit_B else "✗ ФАКТОР B (ЧАСТИНИ) НЕЗНАЧУЩИЙ")

    # =========================================================================
    # 4. КОРЕЛЯЦІЙНИЙ АНАЛІЗ
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. КОРЕЛЯЦІЙНИЙ АНАЛІЗ")
    print("=" * 70)

    # Крок 1-2: Нормалізація та кореляційна матриця
    print("\n4.1-4.2. Нормалізація та кореляційна матриця")
    n_samples = normalized_data.shape[0]
    correlation_matrix = (1 / n_samples) * np.dot(normalized_data.T, normalized_data)

    # Візуалізація кореляційної матриці
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Коефіцієнт кореляції')
    plt.title('Кореляційна матриця 12 каналів ЕКГ', fontsize=14)
    plt.xticks(ticks=np.arange(12), labels=[f'{i}' for i in range(12)])
    plt.yticks(ticks=np.arange(12), labels=[f'{i}' for i in range(12)])

    for i in range(12):
        for j in range(12):
            plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', ha="center", va="center",
                     color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black", fontsize=8)

    plt.tight_layout()
    plt.show()

    # Крок 3: Вибір параметрів для аналізу
    print("\n4.3. Вибір параметрів для аналізу")

    # Знаходимо пари з високою кореляцією
    high_corr_pairs = []
    for i in range(12):
        for j in range(i + 1, 12):
            corr_val = abs(correlation_matrix[i, j])
            if corr_val > 0.7:
                high_corr_pairs.append((i, j, corr_val))

    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

    print("Пари з найвищою кореляцією (|r| > 0.7):")
    for i, j, corr in high_corr_pairs[:5]:
        print(f"  Канал {i} - Канал {j}: r = {corr:.3f}")

    # Вибір груп параметрів
    a, b, c = 0, 1, 2  # Спрощений вибір для демонстрації
    d = 3
    print(f"\nОбрано параметри для аналізу: a={a}, b={b}, c={c}, d={d}")

    # Крок 4-7: Часткові та множинні коефіцієнти кореляції
    print("\n4.4-4.7. Часткові та множинні коефіцієнти кореляції")

    # Часткові коефіцієнти
    r_ab_c = partial_correlation(correlation_matrix[a, b], correlation_matrix[a, c], correlation_matrix[b, c])
    r_ac_b = partial_correlation(correlation_matrix[a, c], correlation_matrix[a, b], correlation_matrix[b, c])

    # Множинні коефіцієнти
    r_a_bc = multiple_correlation_2var(correlation_matrix[a, b], correlation_matrix[a, c], correlation_matrix[b, c])
    R_a_bcd = multiple_correlation_3var(correlation_matrix[a, b], r_ac_b, r_ab_c)

    print(f"r_ab(c) = {r_ab_c:.4f}")
    print(f"r_ac(b) = {r_ac_b:.4f}")
    print(f"r_a/bc = {r_a_bc:.4f}")
    print(f"R_a/bcd = {R_a_bcd:.4f}")

    # Крок 8: Висновки
    print("\n4.8. Висновки про кореляцію між параметрами")

    # Аналіз незалежних параметрів
    mean_correlations = [(i, np.mean(np.abs([correlation_matrix[i, j] for j in range(12) if j != i])))
                         for i in range(12)]
    mean_correlations.sort(key=lambda x: x[1])

    print("\nНайбільш незалежні параметри:")
    for i, (channel, mean_corr) in enumerate(mean_correlations[:3]):
        print(f"  {i + 1}. Канал {channel}: середня |r| = {mean_corr:.3f}")

    # =========================================================================
    # 5. ПЕРЕТВОРЕННЯ ФУР'Є
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. ПЕРЕТВОРЕННЯ ФУР'Є")
    print("=" * 70)

    print("\n5.1. Виконання перетворення Фур'є для кожного каналу")

    fourier_results = []
    reconstructed_signals = np.zeros_like(ecg_data)

    # Параметри частот (винесено за межі циклу)
    sampling_rate = 500  # Гц (500 точок за секунду)

    for channel in range(12):
        signal = ecg_data[:, channel]
        N = len(signal)

        # Пряме та обернене перетворення Фур'є
        A, B, C = fourier_transform(signal)
        reconstructed = inverse_fourier_transform(A, B, N)
        reconstructed_signals[:, channel] = reconstructed

        # Параметри частот
        freq_step = sampling_rate / N

        fourier_results.append({
            'channel': channel, 'A': A, 'B': B, 'C': C,
            'freq_step': freq_step, 'first_freq': freq_step
        })

        if channel < 3:  # Вивід тільки для перших 3 каналів
            print(
                f"Канал {channel + 1}: N={N}, Δf={freq_step:.4f} Гц, f₁={freq_step:.4f} Гц, f_max={sampling_rate / 2:.1f} Гц")

    # Візуалізація спектрів
    print("\n5.2-5.3. Візуалізація спектрів")

    for plot_type in ['full', 'partial']:
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i in range(min(6, len(fourier_results))):
            C = fourier_results[i]['C']
            n_points = len(C) if plot_type == 'full' else min(200, len(C))

            axes[i].plot(C[:n_points], linewidth=1)
            axes[i].set_title(f'Спектр каналу {i + 1}' +
                              (f' (перші {n_points} точок)' if plot_type == 'partial' else ''))
            axes[i].set_xlabel('Частота (індекс)')
            axes[i].set_ylabel('Амплітуда C_j')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Порівняння сигналів
    print("\n5.4. Порівняння початкових та реконструйованих сигналів")

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    for i in range(3):
        original = ecg_data[:, i]
        reconstructed = reconstructed_signals[:, i]

        mse = np.mean((original - reconstructed) ** 2)
        correlation = np.corrcoef(original, reconstructed)[0, 1]

        axes[i].plot(original[:1000], label='Оригінал', linewidth=1, alpha=0.8)
        axes[i].plot(reconstructed[:1000], label='Реконструкція', linewidth=1, alpha=0.8)
        axes[i].set_title(f'Канал {i + 1}: MSE = {mse:.6f}, Корреляція = {correlation:.6f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Аналіз якості реконструкції
    print("\n5.5. Аналіз якості реконструкції")

    reconstruction_stats = []
    for i in range(12):
        original = ecg_data[:, i]
        reconstructed = reconstructed_signals[:, i]

        mse = np.mean((original - reconstructed) ** 2)
        correlation = np.corrcoef(original, reconstructed)[0, 1]
        snr = 10 * np.log10(np.var(original) / mse) if mse > 0 else float('inf')

        reconstruction_stats.append({
            'channel': i + 1, 'mse': mse, 'correlation': correlation, 'snr_db': snr
        })

    # Вивід результатів для перших 3 каналів
    for stat in reconstruction_stats[:3]:
        print(f"Канал {stat['channel']}: MSE={stat['mse']:.8f}, "
              f"Корреляція={stat['correlation']:.6f}, SNR={stat['snr_db']:.2f} dB")

    # Аналіз спектральних характеристик
    print("\n5.6. Аналіз спектральних характеристик")

    for i in range(min(3, len(fourier_results))):
        C = fourier_results[i]['C']
        freq_step = fourier_results[i]['freq_step']

        # Знаходимо піки
        peaks = []
        for j in range(1, len(C) - 1):
            if C[j] > C[j - 1] and C[j] > C[j + 1] and C[j] > np.mean(C) * 1.5:
                peaks.append((j, C[j], j * freq_step))

        peaks.sort(key=lambda x: x[1], reverse=True)

        print(f"\nКанал {i + 1}:")
        for idx, (_, amplitude, freq_hz) in enumerate(peaks[:3]):
            print(f"  Пік {idx + 1}: {freq_hz:.2f} Гц (амплітуда: {amplitude:.4f})")

    # =========================================================================
    # ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
    # =========================================================================
    print("\n" + "=" * 70)
    print("ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")
    print("=" * 70)

    # Зберігаємо основні результати
    results_files = {
        'statistics_results.csv': statistics_df,
        'normalized_ecg_data.csv': normalized_data,
        'two_factor_table.csv': two_factor_df,
        'correlation_matrix.csv': pd.DataFrame(correlation_matrix),
        'reconstructed_signals.csv': reconstructed_signals
    }

    for filename, data in results_files.items():
        if isinstance(data, pd.DataFrame):
            data.to_csv(filename, encoding='utf-8')
        else:
            np.savetxt(filename, data, delimiter=',')
        print(f"✓ {filename}")

    print("\n" + "=" * 70)
    print("ВСІ ЗАВДАННЯ ВИКОНАНО УСПІШНО!")
    print("=" * 70)


if __name__ == "__main__":
    main()
