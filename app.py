# app.py

"""
Веб-приложение на Gradio для интерактивной демонстрации инференса модели.

Запускает веб-сервер, который позволяет пользователю:
- Выполнять предсказание для случайного R-пика.
- (В будущем) Выбирать конкретного пациента и R-пик для анализа.

Перед запуском интерфейса скрипт проверяет наличие файла со статистикой
и при необходимости генерирует его.

Для запуска: `python app.py`
"""

import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from src.config import config
from src.window_inferencer import WindowInference
from src.dataset_preprocessing import DatasetPreprocessing

#########################################################################
# Подготовительный этап перед запуском GUI
#########################################################################

# 1. Убедимся, что JSON-файл со статистикой существует
print("Проверка наличия файла сводки по пациентам...")
preprocessor_for_summary = DatasetPreprocessing()
preprocessor_for_summary.ensure_patient_summary_exists()

# 2. Инициализируем инференсер, который теперь точно найдет этот JSON
print("Инициализация инференсера...")
inferencer = WindowInference(prefix='top')

#########################################################################
# Бэкенд-функции для Gradio
#########################################################################

def run_random_prediction() -> tuple[Figure, dict, str]:
    """
    Запускает инференс для случайного R-пика и форматирует результат для GUI.

    Вызывает метод `predict_random` из инференсера, на основе полученных данных
    генерирует график matplotlib, словарь для компонента gr.Label и
    форматированную строку с деталями предсказания.

    :return: Кортеж из трех элементов: (matplotlib.figure.Figure, dict, str)
    """
    print("Запущен инференс для случайного пика...")
    
    # 1. Получаем результаты от бэкенда
    results = inferencer.predict_random()
    
    # 2. Форматируем график (matplotlib figure)
    fig = plt.figure(figsize=(10, 3))
    plt.plot(results['window_data'].flatten()) # .flatten() на случай, если данные многомерные
    plt.title(f"Пациент: {results['patient_id']}, Сэмпл: {results['sample_id']}")
    plt.xlabel("Отсчеты")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.tight_layout() 
    
    # 3. Форматируем метку с вердиктом
    pred_s1 = results['prediction_s1']
    pred_s1_display = inferencer.display_names.get(pred_s1, pred_s1)

    final_prediction_display = pred_s1_display
    if 'prediction_s2' in results:
        pred_s2 = results['prediction_s2']
        pred_s2_display = inferencer.display_names.get(pred_s2, pred_s2)
        final_prediction_display = f"{pred_s1_display} -> {pred_s2_display}"
        
    verdict = {final_prediction_display: 1.0} # Формат для gr.Label: {метка: уверенность}
    
    # 4. Форматируем текстовые детали
    true_s1_display = inferencer.display_names.get(results['true_label_s1'], results['true_label_s1'])

    details_text = f"Истинная метка (Stage 1): {true_s1_display}\n"
    details_text += f"Предсказание (Stage 1): {pred_s1_display} (Уверенность: {results['confidence_s1']:.2f}%)\n"
    if 'prediction_s2' in results:
        true_s2_display = inferencer.display_names.get(results['true_label_s2'], results['true_label_s2'])
        pred_s2_display = inferencer.display_names.get(results['prediction_s2'], results['prediction_s2'])
    
        details_text += "------\n"
        details_text += f"Истинная метка (Stage 2): {true_s2_display}\n"
        details_text += f"Предсказание (Stage 2): {pred_s2_display} (Уверенность: {results['confidence_s2']:.2f}%)"

    return fig, verdict, details_text

def update_on_patient_select(formatted_patient_str: str) -> str:
    """
    Вызывается при выборе пациента в Dropdown.
    Возвращает детализированную статистику по пациенту в виде markdown-строки.

    :param formatted_patient_str: Строка, выбранная в Dropdown (напр., "ID 101 [...]").
    :return: Отформатированная markdown-строка со статистикой.
    """
    return inferencer.get_patient_stats_markdown(formatted_patient_str)

def run_prediction(formatted_patient_str: str, sample_id: int) -> tuple[Figure | None, dict, str]:
    """
    Запускает инференс для конкретного R-пика по выбору пользователя.
    Обрабатывает ввод, вызывает инференсер и форматирует результат для GUI.

    :param formatted_patient_str: Строка из Dropdown с ID пациента.
    :param sample_id: Номер сэмпла R-пика, введенный пользователем.
    :return: Кортеж (Figure | None, dict, str) для обновления компонентов вывода.
    """
    # Валидация входных данных
    if not formatted_patient_str or not sample_id:
        return None, {}, "Ошибка: Выберите пациента и введите номер сэмпла."
    
    try:
        patient_id = formatted_patient_str.split(' ')[1]
        sample_id = int(sample_id)
    except (IndexError, ValueError):
        return None, {}, "Ошибка: Неверный формат ID пациента или сэмпла."
        
    print(f"Запущен инференс для: Пациент {patient_id}, Сэмпл {sample_id}")
    
    # Основная логика
    try:
        results = inferencer.predict_by_id(patient_id, sample_id)
    except ValueError as e:
        return None, {}, f"Ошибка инференса: {e}"

    # Форматирование вывода (аналогично run_random_prediction)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(results['window_data'].flatten())
    ax.set_title(f"Пациент: {results['patient_id']}, Сэмпл: {results['sample_id']}")
    ax.set_xlabel("Отсчеты")
    ax.set_ylabel("Амплитуда")
    ax.grid(True)
    plt.tight_layout() 
    
    pred_s1 = results['prediction_s1']
    pred_s1_display = inferencer.display_names.get(pred_s1, pred_s1)
    final_prediction_display = pred_s1_display
    if 'prediction_s2' in results:
        pred_s2 = results['prediction_s2']
        pred_s2_display = inferencer.display_names.get(pred_s2, pred_s2)
        final_prediction_display = f"{pred_s1_display} -> {pred_s2_display}"
        
    verdict = {final_prediction_display: 1.0}
    
    true_s1_display = inferencer.display_names.get(results['true_label_s1'], results['true_label_s1'])
    details_text = f"Истинная метка (Stage 1): {true_s1_display}\n"
    details_text += f"Предсказание (Stage 1): {pred_s1_display} (Уверенность: {results['confidence_s1']:.2f}%)\n"
    if 'prediction_s2' in results:
        true_s2_display = inferencer.display_names.get(results['true_label_s2'], results['true_label_s2'])
        pred_s2_display = inferencer.display_names.get(results['prediction_s2'], results['prediction_s2'])
        details_text += "------\n"
        details_text += f"Истинная метка (Stage 2): {true_s2_display}\n"
        details_text += f"Предсказание (Stage 2): {pred_s2_display} (Уверенность: {results['confidence_s2']:.2f}%)"

    return fig, verdict, details_text

def handle_show_region_button(formatted_patient_str: str, center_percent: float) -> tuple[Figure | None, dict, dict]:
    """
    Обработчик кнопки "Показать участок". Загружает данные в буфер.
    Возвращает первую отрисовку графика и обновленные параметры для слайдеров.
    """
    if not formatted_patient_str:
        # Возвращаем пустой график и скрываем слайдер прокрутки
        return None, gr.State(value=None), gr.Slider(visible=False)

    patient_id = formatted_patient_str.split(' ')[1]
    
    # Вызываем метод из бэкенда для получения данных
    region_data = inferencer.get_ecg_region_data(patient_id, center_percent)
    
    if region_data is None:
        return None, gr.State(value=None), gr.Slider(visible=False)

    # Рисуем первую версию графика (центральную часть)
    fig = redraw_region_plot(region_data, 50) # 50% - центр буфера

    # Обновляем слайдер прокрутки: делаем его видимым
    scroll_slider_update = gr.Slider(visible=True)

    return fig, region_data, scroll_slider_update

def redraw_region_plot(region_data: dict, scroll_percent: float) -> Figure | None:
    """
    Перерисовывает график участка ЭКГ на основе данных из буфера и положения слайдера прокрутки.
    """
    if region_data is None:
        return None

    signal_df = region_data["signal_df"]
    peaks = region_data["peaks"]

    # Определяем, какую часть буфера показывать
    view_width_samples = config['data']['visualization']['view_width_samples']

    # Общая длина буфера в сэмплах
    buffer_start_sample = signal_df['Sample'].iloc[0]
    buffer_end_sample = signal_df['Sample'].iloc[-1]
    total_buffer_samples = buffer_end_sample - buffer_start_sample
###     total_buffer_samples = len(signal_df)

    # Определяем центральную точку для нашего "окна" на основе положения слайдера
    center_view_sample = buffer_start_sample + int(total_buffer_samples * (scroll_percent / 100))
###     center_view_sample = int(total_buffer_samples * (scroll_percent / 100))
    
    # Рассчитываем границы "окна"
    view_start_abs = max(buffer_start_sample, center_view_sample - view_width_samples // 2)
    view_end_abs = min(buffer_end_sample, center_view_sample + view_width_samples // 2)

    # Отрисовка
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Рисуем сигнал только в видимой области
    view_df = signal_df[(signal_df['Sample'] >= view_start_abs) & (signal_df['Sample'] <= view_end_abs)]
    ax.plot(view_df['Sample'], view_df['Signal'])

    # Цвета для выборок
    split_colors = {'TRAIN': 'grey', 'VAL': 'blue', 'TEST': 'green'}
    
    ### # Переменная для чередования высоты текста
    ### text_on_top = True 

    # Определяем 4 уровня высоты для текста, чтобы избежать наложения.
    # Значения подобраны так, чтобы текст не выходил за верхнюю границу графика.
    y_max = ax.get_ylim()[1]
    y_levels = [y_max * 1.0, y_max * 0.9, y_max * 0.8, y_max * 0.7]
    
    # Для эффективности сначала отфильтруем только те пики, что видны на экране.
    visible_peaks = [p for p in peaks if view_start_abs <= p['sample'] <= view_end_abs]

    # Рисуем аннотации R-пиков
    ### for peak in peaks:
    for i, peak in enumerate(visible_peaks):

        ### if view_start_abs <= peak['sample'] <= view_end_abs:
            split = peak.get('split', 'UNKNOWN').upper()
            color = split_colors.get(split, 'black')

            # Используем сокращения для N на графике
            peak_label = peak.get('type', '?')
            if peak_label == 'N+N':
                peak_label = 'N+'
            elif peak_label == 'N (по Aux не N)':
                peak_label = 'N-'

            # Рисуем линию
            ax.axvline(x=peak['sample'], color=color, linestyle='--', linewidth=0.8)

            # Чередуем высоту текста по 4 уровням, используя остаток от деления индекса пика.
            y_pos = y_levels[i % len(y_levels)]

            ### # Размещаем текст чуть выше средней линии графика, чтобы не мешать заголовку вкладки
            ### y_pos = np.mean(ax.get_ylim()) + (ax.get_ylim()[1] - np.mean(ax.get_ylim())) * 0.8
                        
            # Логика отрисовки текста
            ax.text(
                x=peak['sample'],
                y=y_pos, 
                s=f" {peak_label} ({peak['sample']})", # Формат " ТИП (ID)"
                color=color,
                fontsize=12,
                fontweight='bold',
                ha='center', # Горизонтальное выравнивание
                va='bottom' # Вертикальное выравнивание
            )

    # Заголовок
    ax.set_xlabel("Сэмпл")
    ax.set_ylabel("Амплитуда")
    ax.grid(True)

    # Увеличиваем верхний предел оси Y на 15%, чтобы текст не накладывался на заголовок.
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], current_ylim[1] * 1.15)
    
    # Расширяем пределы оси X, чтобы крайние метки не обрезались.
    # Добавляем поля, равные 1% от ширины видимого окна.
    current_xlim = ax.get_xlim()
    xlim_range = current_xlim[1] - current_xlim[0]
    ax.set_xlim(current_xlim[0] - xlim_range * 0.01, current_xlim[1] + xlim_range * 0.01)

    # Легенда
    legend_elements = [
        Line2D([0], [0], color='grey', lw=2, label='Train'),
        Line2D([0], [0], color='blue', lw=2, label='Validation'),
        Line2D([0], [0], color='green', lw=2, label='Test')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize='small')

    plt.tight_layout()

    return fig

#########################################################################
# Создание интерфейса Gradio
#########################################################################

### # CSS-стиль для увеличения иконки внутри кнопки с классом 'big-icon-button':
### # - Увеличивает иконку до 32px.
### # - Смещает все содержимое кнопки влево.
### # - Добавляет отступ слева для красоты.
### custom_css = """
### .big-icon-button img { width: 32px !important; height: 32px !important; }
### .big-icon-button { justify-content: flex-start !important; padding-left: 10px !important; }
### """

# Загружаем стили из внешнего CSS-файла, чтобы не засорять код.
try:
    with open("assets/style.css", "r", encoding="utf-8") as f:
        custom_css = f.read()
except FileNotFoundError:
    print("ПРЕДУПРЕЖДЕНИЕ: Файл assets/style.css не найден. Интерфейс будет отображаться без кастомных стилей.")
    custom_css = ""
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="cyan"), css=custom_css) as demo:

    # Создаем State для хранения буфера данных
    region_data_buffer = gr.State(value=None)

    gr.Markdown("# Анализатор R-пиков ЭКГ")
    
    with gr.Row():
        # Левая колонка (Управление)
        with gr.Column(scale=1):

            gr.Markdown("### Случайный пример")
            gr.Markdown("Нажмите кнопку, чтобы получить предсказание для случайного R-пика из тестовой выборки.")
            random_button = gr.Button(
                "Получить случайный пример",
                icon="assets/med_icon_1.png",
                elem_classes=["big-icon-button"]
            )
            
            gr.Markdown("---")
            gr.Markdown("### Анализ конкретного R-пика")
            patient_dropdown = gr.Dropdown(
                choices=inferencer.formatted_patient_list,
                label="1. Выберите пациента",
                info="В списке показана статистика по основным типам аномалий у пациента."
            )

            patient_stats_display = gr.Markdown("Выберите пациента, чтобы увидеть подробную статистику.")
 
            # Оборачиваем визуализатор в gr.Accordion для компактности
            with gr.Accordion("2. Визуальный выбор R-пика", open=True):
                region_slider = gr.Slider( # Слайдер "Обзор"
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Обзор записи пациента (%)"
                )
                show_region_button = gr.Button(
                    "Показать участок",
                    icon="assets/med_icon_2.png",
                    elem_classes=["big-icon-button"]
                )
                
                ### region_plot = gr.Plot(label="Участок кардиограммы") # Plot для участка
                
                scroll_slider = gr.Slider( # Слайдер "Прокрутка", изначально невидимый
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Прокрутка участка",
                    visible=False 
                )
            
            gr.Markdown("---")
            
            ### sample_input = gr.Textbox(label="2. Введите номер сэмпла R-пика", info="Скопируйте номер из статистики выше или введите свой.", elem_id="sample_input_box")
            sample_input = gr.Textbox(label="3. Введите номер сэмпла R-пика", info="Найдите интересный пик на графике выше и введите его номер сюда.")

            analyze_button = gr.Button(
                "Анализ R-пика",
                icon="assets/med_icon_3.png",
                elem_classes=["big-icon-button"]
                )

        # Правая колонка (Результаты)
        with gr.Column(scale=2):

        # Система вкладок
            with gr.Tabs():
                # Вкладка для основного анализа
                with gr.Tab(label="Результат анализа"):
                    output_plot = gr.Plot(label="Сигнал R-пика")
                    output_label = gr.Label(label="Вердикт")
                    output_details = gr.Textbox(label="Детали предсказания", lines=4)
                
                # Вкладка для визуального обзора
                with gr.Tab(label="Обзор участка"):
                    region_plot = gr.Plot(label="Участок кардиограммы")

    # Связывание логики и интерфейса
    random_button.click(
        fn=run_random_prediction,
        inputs=[],
        outputs=[output_plot, output_label, output_details]
    )
    patient_dropdown.change(
        fn=update_on_patient_select,
        inputs=[patient_dropdown],
        outputs=[patient_stats_display]
    )
    analyze_button.click(
        fn=run_prediction,
        inputs=[patient_dropdown, sample_input],
        outputs=[output_plot, output_label, output_details]
    )
    show_region_button.click(
        fn=handle_show_region_button,
        inputs=[patient_dropdown, region_slider],
        outputs=[region_plot, region_data_buffer, scroll_slider]
    )
    scroll_slider.input(
        fn=redraw_region_plot,
        inputs=[region_data_buffer, scroll_slider],
        outputs=[region_plot]
    )

#########################################################################
# Запуск приложения
#########################################################################

if __name__ == "__main__":
    demo.launch()