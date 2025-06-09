# fashion-category-prediction

Данный проект позволяет распознавать тип одежды по его закодированному изображению 28x28.

## Установка

Клонируйте репозиторий:

```bash
git clone https://github.com/Parenbz/fashion-category-prediction.git
cd fashion-category-prediction
```

Установите зависимости:

```bash
poetry install
```

Загрузите данные для обучения модели (в качестве удалённого DVC репозитория используется .dvc/tmp):

```bash
poetry run dvc pull
```

Настройте пре-коммит и проверьте, что всё корректно:

```bash
poetry run pre-commit install
poetry run pre-commit run -a
```

## Использование

### Обучение модели

Все гиперпараметры для обучения настраиваются из файла 'configs/train/train.yaml' с помощью Hydra. Начать обучение можно командой

```bash
poetry run python train.py
```

### Инференс

Изображения 28x28 кодируются в строку из чисел 1-255, обозначающих черноту каждого пикселя, и такие строки помещаются в файл input.csv. Модель работает по команде

```bash
poetry run python infer.py
```

### Метрики

Во время обучения модели сохраняются графики в папку plots, и просмотреть их можно по команде

```bash
poetry run tensorboard --logdir=plots
```
