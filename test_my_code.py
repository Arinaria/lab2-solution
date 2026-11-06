from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput
from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations

print("Тестируем Titanic")
titanic_data = TitanicInput(csv_path='titanic.csv')
titanic_res = analyze_titanic(titanic_data)

print(f"Всего пассажиров: {titanic_res.row_count}")
print(f"Пассажиров старше 30: {titanic_res.adults_over_30_count}")
print(f"Топ-5 тарифов: {titanic_res.highest_fares}")

print("Тестируем Brain Size")
brain_data = BrainDataInput(csv_path='brainsize.txt')
brain_res = analyze_brain_correlations(brain_data)

print(f"Мужчин: {brain_res.men_count}, Женщин: {brain_res.women_count}")
print(f"Сильнейшая корреляция с MRI у признака: {brain_res.strongest_mri_feature}")