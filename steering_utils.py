from tqdm.auto import tqdm
from controller import VectorStore, register_vector_control
from collections import defaultdict
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_model(pipe, prompt, seed, num_denoising_steps, device=DEVICE):
    """Запуск модели для генерации изображения"""
    image = pipe(
        prompt=prompt, 
        num_inference_steps=num_denoising_steps, 
        generator=torch.Generator(device=device).manual_seed(seed)
    ).images[0]

    return image

    
def get_imagenet_classes(num):
    """Получение разнообразных префиксов для генерации изображений"""
    # Список разнообразных префиксов
    diverse_prefixes = [
        "cyberpunk", "watercolor painting", "steampunk", "vaporwave", "minimalist", 
        "hyperrealistic", "ancient egyptian", "clay sculpture", "pixel art", "baroque", 
        "comic book", "glitch art", "ukiyo-e", "cubism", "gothic", 
        "microscopic", "cosmic nebula", "dia de los muertos", "blueprint", "stained glass", 
        "synthwave", "industrial", "art nouveau", "brutalist", "pop art", 
        "impressionist", "surrealist", "anime", "retro 50s", "film noir", 
        "medieval manuscript", "mosaic", "paper craft", "graffiti", "low poly 3D", 
        "oil painting", "neon", "pastel", "dystopian", "children's book illustration", 
        "pencil sketch", "chiaroscuro", "art deco", "isometric", "pointillism", 
        "propaganda poster", "woodcut", "embroidery", "cave painting", "holographic"
    ]
    return diverse_prefixes[:num]


def get_prompts_concrete(num=20, concept_pos=None, concept_neg=None):
    """Генерация промптов с конкретными объектами"""
    assert (concept_pos is not None), "concept_pos не должен быть None!"
    
    imagenet_classes = get_imagenet_classes(num)
    prompts_pos = []
    prompts_neg = []
    for cls in imagenet_classes[:num]:
        prompts_pos.append(cls+' with {}'.format(concept_pos))
        if concept_neg is not None:
            prompts_neg.append(cls+' with {}'.format(concept_neg))
        else:
            prompts_neg.append(cls)

    if concept_neg is not None:
        assert len(prompts_pos[:num]) == len(prompts_neg[:num])
        
    return prompts_pos[:num], prompts_neg[:num]


def compute_steering_vectors(
    pipe,
    device=DEVICE,
    concept_pos=None, 
    concept_neg=None,
    num_denoising_steps=50,
    max_prompts=None,
    n_times=1  # количество запусков одного и того же промпта
):
    """
    Вычисляет управляющие векторы для указанной модели и концепции

    Возвращает словарь векторов управления.
    Для каждого набора промптов выполняется генерация с разными seed'ами n_times раз.
    """
    print(f"Вычисляем управляющие векторы...")

    prompts_pos, prompts_neg = get_prompts_concrete(
        num=max_prompts,
        concept_pos=concept_pos,
        concept_neg=concept_neg
    )

    # Вычисляем выходы CA для промптов
    pos_vectors = []
    neg_vectors = []

    print(f"Обрабатываем {len(prompts_pos)} пар промптов, по {n_times} запусков для каждой...")

    # Проходим по каждой паре промптов
    for i, (prompt_pos, prompt_neg) in tqdm(enumerate(zip(prompts_pos, prompts_neg)), total=len(prompts_pos)):
        
        # Запускаем генерацию для одного промпта n_times раз с разными seed'ами
        for t in range(n_times):
            current_seed = i * n_times + t  # генерация уникального seed для каждого прогона
            print(f"Промпт {i+1}/{len(prompts_pos)}, запуск {t+1}/{n_times}: '{prompt_pos}' и '{prompt_neg}'")

            # Для положительного промпта
            controller = VectorStore(steer=False, device=device)
            register_vector_control(pipe.unet, controller)
            image = run_model(pipe, prompt_pos, current_seed, num_denoising_steps, device)
            pos_vectors.append(controller.vector_store)

            # Для отрицательного промпта
            controller = VectorStore(steer=False, device=device)
            register_vector_control(pipe.unet, controller)
            image = run_model(pipe, prompt_neg, current_seed, num_denoising_steps, device)
            neg_vectors.append(controller.vector_store)

    # Вычисляем управляющие векторы
    steering_vectors = {}

    # Предполагается, что количество прогонов теперь равно len(prompts_pos) * n_times
    for denoising_step in range(num_denoising_steps):
        steering_vectors[denoising_step] = defaultdict(list)

        for key in ['up', 'down', 'mid']:
            # Считаем, что структура в pos_vectors и neg_vectors одинакова,
            # и берем количество слоев по первому элементу
            num_layers = len(pos_vectors[0][denoising_step][key])
            for layer_num in range(num_layers):
                
                # Собираем векторы для текущего слоя
                pos_vectors_layer = [run_vectors[denoising_step][key][layer_num] for run_vectors in pos_vectors]
                pos_vectors_avg = np.mean(pos_vectors_layer, axis=0)

                neg_vectors_layer = [run_vectors[denoising_step][key][layer_num] for run_vectors in neg_vectors]
                neg_vectors_avg = np.mean(neg_vectors_layer, axis=0)

                # Вычисляем и нормализуем управляющий вектор
                steering_vector = pos_vectors_avg - neg_vectors_avg
                steering_vector = steering_vector / np.linalg.norm(steering_vector)

                steering_vectors[denoising_step][key].append(steering_vector)
    
    print("Управляющие векторы успешно вычислены!")
    return steering_vectors


def generate_image(
    pipe,
    prompt,
    device=DEVICE,
    steering_vectors=None,
    seed=0,
    alpha=10,
    beta=1.0,
    steer_only_up=False,
    steer_back=False,
    not_steer=False,
    num_denoising_steps=50,
):
    """
    Генерирует изображение с использованием указанной модели и векторов управления
    
    Возвращает сгенерированное изображение
    """
    
    # Проверяем, что число шагов соответствует доступным векторам
    if steering_vectors is not None and not not_steer:
        max_steps = max(steering_vectors.keys()) + 1
        if num_denoising_steps > max_steps:
            print(f"Предупреждение: сокращаем количество шагов с {num_denoising_steps} до {max_steps}")
            num_denoising_steps = max_steps

    if not_steer:
        # Генерация без управления
        print(f"Генерируем изображение без управления: промпт='{prompt}'")
        image = run_model(model_name, pipe, prompt, seed, num_denoising_steps, device)
    else:
        # Генерация с управлением
        print(f"Генерируем изображение с управлением: промпт='{prompt}'")

        controller = VectorStore(
            steering_vectors, 
            device=device,
        )
        controller.steer_only_up = steer_only_up

        if steer_back:
            controller.steer_back = True
            controller.beta = beta
            print(f"Режим удаления концепции (beta={beta})")
        else:
            controller.steer_back = False
            controller.alpha = alpha

        register_vector_control(pipe.unet, controller)
        image = run_model(pipe, prompt, seed, num_denoising_steps, device)

    return image