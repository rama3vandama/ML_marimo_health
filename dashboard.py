# Все элементы управления с новыми именами
client_age = mo.ui.slider(18, 70, value=35, step=1, label="Возраст")
client_bmi = mo.ui.slider(15, 50, value=25, step=0.5, label="Индекс массы тела")
client_children = mo.ui.slider(0, 5, value=2, step=1, label="Количество детей")

client_gender = mo.ui.dropdown(
    ["мужчина", "женщина"], 
    value="мужчина", 
    label="Пол"
)

client_smokes = mo.ui.dropdown(
    ["нет", "да"], 
    value="нет", 
    label="Курит"
)

insurance_plan = mo.ui.dropdown(
    ["Basic", "Standard", "Premium"], 
    value="Standard", 
    label="Уровень страховки"
)

client_region = mo.ui.dropdown(
    ["southwest", "southeast", "northwest", "northeast"], 
    value="southwest", 
    label="Регион проживания"
)

client_job = mo.ui.dropdown(
    ["White collar", "Blue collar", "Student"], 
    value="White collar", 
    label="Род занятий"
)

medical_background = mo.ui.dropdown(
    ["No History", "High blood pressure", "Diabetes", "Heart disease"], 
    value="No History", 
    label="Медицинская история"
)

family_medical = mo.ui.dropdown(
    ["No Family History", "High blood pressure", "Diabetes", "Heart disease"], 
    value="No Family History", 
    label="Семейная история болезней"
)

activity_level = mo.ui.dropdown(
    ["None", "Light", "Moderate", "Heavy"], 
    value="Moderate", 
    label="Уровень физической активности"
)

# Интерфейс с элементами управления
mo.vstack([
    mo.md("# 🎛️ Параметры страхового полиса"),
    
    mo.md("## 👤 Демографические данные"),
    mo.hstack([
        mo.vstack([client_age, client_bmi]),
        mo.vstack([client_children, client_gender])
    ]),
    
    mo.md("## 🏥 Состояние здоровья"),
    mo.hstack([
        mo.vstack([client_smokes, medical_background]),
        mo.vstack([family_medical, activity_level])
    ]),
    
    mo.md("## 📍 Дополнительная информация"),
    mo.hstack([
        insurance_plan,
        client_region,
        client_job
    ])
])

# Словари для преобразования
gender_conversion = {"мужчина": 0, "женщина": 1}
smoker_conversion = {"нет": 0, "да": 1}
coverage_conversion = {"Basic": 0, "Standard": 1, "Premium": 2}
medical_conversion = {"No History": 0, "High blood pressure": 1, "Diabetes": 2, "Heart disease": 3}
# Для семейной истории используем тот же словарь
family_medical_conversion = {"No Family History": 0, "High blood pressure": 1, "Diabetes": 2, "Heart disease": 3}
activity_conversion = {"None": 0, "Light": 1, "Moderate": 2, "Heavy": 3}

@mo.cache
def calculate_insurance_cost(
    age_val, bmi_val, children_val, gender_val, smoker_val, coverage_val, 
    region_val, job_val, medical_val, family_medical_val, activity_val
):
    # Создаем словарь с входными данными
    input_data = {
        'age': age_val,
        'bmi': bmi_val,
        'children': children_val,
        'gender': gender_conversion[gender_val],
        'smoker': smoker_conversion[smoker_val],
        'coverage_level': coverage_conversion[coverage_val],
        'region': encoders_dict['region'].transform([region_val])[0],
        'occupation': encoders_dict['occupation'].transform([job_val])[0],
        'medical_history': medical_conversion[medical_val],
        'family_medical_history': family_medical_conversion[family_medical_val],  # Используем правильный словарь
        'exercise_frequency': activity_conversion[activity_val]
    }
    
    # Создаем DataFrame в правильном порядке
    input_dataframe = pd.DataFrame([input_data])[all_feature_names]
    
    # Делаем предсказание
    predicted_cost = insurance_model.predict(xgb.DMatrix(input_dataframe))[0]
    
    # Форматируем результат
    cost_formatted = f"${predicted_cost:,.2f}"
    cost_rounded = f"${predicted_cost:,.0f}"
    
    return mo.vstack([
        mo.md("## 💰 Расчет стоимости страховки"),
        mo.md(f"# {cost_rounded}"),
        mo.md(f"*Точная сумма: {cost_formatted}*"),
        
        mo.md("---"),
        
        mo.md("### 📋 Введенные параметры:"),
        mo.vstack([
            mo.hstack([
                mo.md(f"**Возраст:** {age_val} лет"),
                mo.md(f"**BMI:** {bmi_val:.1f}"),
                mo.md(f"**Дети:** {children_val}")
            ]),
            mo.hstack([
                mo.md(f"**Пол:** {gender_val}"),
                mo.md(f"**Курение:** {smoker_val}"),
                mo.md(f"**Страховка:** {coverage_val}")
            ]),
            mo.hstack([
                mo.md(f"**Регион:** {region_val}"),
                mo.md(f"**Работа:** {job_val}"),
                mo.md(f"**Мед. история:** {medical_val}")
            ]),
            mo.hstack([
                mo.md(f"**Сем. история:** {family_medical_val}"),
                mo.md(f"**Активность:** {activity_val}")
            ])
        ]),
        
        mo.md("---"),
        
        mo.md("### 📊 Информация о модели:"),
        mo.md(f"- Точность модели (R²): {r2_score_value:.4f}"),
        mo.md(f"- Средняя ошибка: ${mae_score:,.2f}"),
        mo.md(f"- Относительная точность: ±{(mae_score/y_test_data.mean()*100):.1f}%")
    ])

# Вызываем функцию расчета
calculate_insurance_cost(
    client_age.value,
    client_bmi.value,
    client_children.value,
    client_gender.value,
    client_smokes.value,
    insurance_plan.value,
    client_region.value,
    client_job.value,
    medical_background.value,
    family_medical.value,
    activity_level.value
)


max_depth_slider = mo.ui.slider(3, 15, value=8, step=1, label="Макс. глубина деревьев (max_depth)")
learning_rate_slider = mo.ui.slider(0.01, 0.5, value=0.1, step=0.01, label="Скорость обучения (learning_rate)")
subsample_slider = mo.ui.slider(0.1, 1.0, value=0.8, step=0.05, label="Доля выборки (subsample)")
colsample_slider = mo.ui.slider(0.1, 1.0, value=0.8, step=0.05, label="Доля признаков (colsample_bytree)")
n_estimators_slider = mo.ui.slider(50, 500, value=200, step=50, label="Количество деревьев (n_estimators)")

mo.vstack([
    mo.md("# 🎛️ Настройка гиперпараметров XGBoost"),
    
    mo.md("## Основные параметры:"),
    mo.hstack([
        mo.vstack([max_depth_slider, learning_rate_slider]),
        mo.vstack([subsample_slider, colsample_slider])
    ]),
    
    mo.md("---"),
    
    mo.md("## Дополнительные параметры:"),
    n_estimators_slider,
    
    mo.md("---"),
    
    mo.md("### Текущие значения будут показаны ниже")
])

mo.vstack([
    mo.md("## 📊 Текущие значения гиперпараметров:"),
    mo.md(f"- max_depth: {max_depth_slider.value}"),
    mo.md(f"- learning_rate: {learning_rate_slider.value:.3f}"),
    mo.md(f"- subsample: {subsample_slider.value:.2f}"),
    mo.md(f"- colsample_bytree: {colsample_slider.value:.2f}"),
    mo.md(f"- n_estimators: {n_estimators_slider.value}")
])

import time

@mo.cache
def train_model_compact(max_depth_val, learning_rate_val, subsample_val, 
                       colsample_val, n_estimators_val):
    
    start_time = time.time()
    
    model_params = {
        'max_depth': int(max_depth_val),
        'learning_rate': learning_rate_val,
        'subsample': subsample_val,
        'colsample_bytree': colsample_val,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Обучение модели
    model = xgb.train(
        model_params,
        train_matrix,
        num_boost_round=int(n_estimators_val),
        verbose_eval=False
    )
    
    # Предсказания и метрики
    test_predictions = model.predict(test_matrix)
    from sklearn.metrics import mean_absolute_error, r2_score
    test_mae = mean_absolute_error(y_test_data, test_predictions)
    test_r2 = r2_score(y_test_data, test_predictions)
    
    training_time = time.time() - start_time
    
    return mo.vstack([
        mo.md("## 🎯 Результаты обучения"),
        
        mo.md("### 📊 Качество модели:"),
        mo.hstack([
            mo.vstack([
                mo.md(f"**MAE:**"),
                mo.md(f"# ${test_mae:,.0f}"),
                mo.md(f"*Средняя абсолютная ошибка*")
            ]),
            mo.vstack([
                mo.md(f"**R²:**"),
                mo.md(f"# {test_r2:.4f}"),
                mo.md(f"*Объясненная дисперсия*")
            ])
        ]),
        
        mo.md("---"),
        
        mo.md("### ⚙️ Использованные гиперпараметры:"),
        mo.vstack([
            mo.md(f"- max_depth: {max_depth_val}"),
            mo.md(f"- learning_rate: {learning_rate_val:.3f}"),
            mo.md(f"- subsample: {subsample_val:.2f}"),
            mo.md(f"- colsample_bytree: {colsample_val:.2f}"),
            mo.md(f"- n_estimators: {n_estimators_val}"),
            mo.md(f"- Время обучения: {training_time:.2f} сек")
        ])
    ])

# Вызываем функцию
train_model_compact(
    max_depth_slider.value,
    learning_rate_slider.value,
    subsample_slider.value,
    colsample_slider.value,
    n_estimators_slider.value
)

# Вызываем функцию
plot_learning_curve(
    max_depth_slider.value,
    learning_rate_slider.value,
    subsample_slider.value,
    colsample_slider.value,
    n_estimators_slider.value
)
