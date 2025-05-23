# **Cross-Attention Steering**

Авторское решение задачи [NEOAI 2025: The Hogspell Challenge](https://www.kaggle.com/competitions/neoai-2025-hogspell-challenge/overview).

> **Примечание.** Имплементация основана на методе из статьи [**CASteer: Steering Diffusion Models for Controllable Generation**](https://arxiv.org/abs/2503.09630).

## Основные идеи

* **Сбор активаций.** Для пары промптов, отличающихся только наличием/отсутствием целевой концепции, сохраняются self‑ и cross‑attention активации каждого `BasicTransformerBlock` в UNet на каждом шаге денойзинга.
* **Построение управляющих векторов.** Агрегируем активации по нескольким сида́м, вычитаем *отрицательные* (без концепции) из *положительных* (с концепцией) и L2‑нормируем, получая по одному вектору на комбинацию «шаг × блок × ветка `{up, mid, down}`».
* **Инъекция во время генерации.** Регистрируем лёгкий контроллер (`VectorStore`), который добавляет `α·v` (или удаляет концепцию при `β·sim`) к скрытому состоянию, сохраняя его норму.

---

## Структура

```text
.
├── controller.py           # рантайм‑хук: хранение/инъекция векторов
├── steering_utils.py       # вспомогательные функции: сбор активаций, построение векторов, генерация
├── steering_baseline.ipynb # подробный ноутбук‑демо
├── steering_vectors.pkl    # пред‑вычисленные векторы для демонстрации
├── prompts.json            # примеры положительных/отрицательных промптов
├── hogs_pictures/          # изображения для README
└── requirements.txt
```

---

## Установка

```bash
# клон
git clone https://github.com/lenjjiv/steering-hogs.git
cd steering-hogs

# (опционально) виртуальное окружение
python -m venv .venv && source .venv/bin/activate

# зависимости
pip install -r requirements.txt
```

---

## Быстрый пример

```python
import torch
from diffusers import StableDiffusionPipeline
from steering_utils import compute_steering_vectors, generate_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1 – загружаем базовый пайплайн
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# 2 – либо берём готовые векторы …
import pickle
steering_vectors = pickle.load(open("steering_vectors.pkl", "rb"))

# … либо строим свои
# steering_vectors = compute_steering_vectors(
#     pipe,
#     concept_pos="pig", # целевая концепция
#     concept_neg="horse", # альтернативная концепция или пустая строка
#     num_denoising_steps=50,
#     max_prompts=20,
#     n_times=2,
# )

# 3 – генерируем!
prompt = "A cyberpunk city street at night, cinematic lighting"
img = generate_image(
    pipe,
    prompt,
    steering_vectors=steering_vectors,
    alpha=12,              # сила steering‑а
    steer_only_up=True,    # меняем только up‑блоки
    num_denoising_steps=50,
    seed=42,
)

img.save("steered.png")
```

---

## Примеры

![Example of steering](hogs_pictures/Public/grid_2_ids_53-54-57-59-60-62-64-65-67-70-74-76-78-79-88-90-91-92-93-94-95-96-101-108-109.png)

---

## Справочник API

| Функция                    | Описание                                                                                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `VectorStore`              | Рантайм‑контроллер, который фиксирует активации **и/или** вводит управляющие векторы. Ключевые аргументы: `alpha`, `beta`, `steer_back`, `steer_only_up`, `device`. |
| `register_vector_control`  | Рекурсивно подключает `VectorStore` к каждому `BasicTransformerBlock` UNet‑а.                                                                                       |
| `compute_steering_vectors` | Строит управляющие векторы по паре положительный/отрицательный промпт.                                                                                              |
| `generate_image`           | Удобная обёртка для генерации с управляющими векторами или без.                                                                                                     |

---

## Эксперименты и ноутбук

В `steering_baseline.ipynb` показано:

1. Строим векторы для концепции **«pig»**.
2. Анализируем нормы и сходство векторов.
3. Генерируем изображения со и без steering‑а.
4. *Удаление концепции* (`steer_back=True, β>0`).

---

## Ограничения и планы

* Проверено только на Stable Diffusion v1.5.
* При больших `α` возможен «перестир» → артефакты; пробуйте аннулирование.
* Сейчас активации хранятся целиком в RAM — стоит перейти на пакетное сохранение при большом числе шагов.
* Нет автоматической метрики; при необходимости добавить CLIPScore / FID.

---

## Благодарности

* Метод основан на работе [**CASteer: Steering Diffusion Models for Controllable Generation**](https://arxiv.org/abs/2503.09630).

---
