# Руководство по настройке FramePack-eichi: Полное руководство по установке для всех сред | [日本語](README_setup.md) | [English](README_setup_en.md) | [繁體中文](README_setup_zh.md)

> **Отказ от ответственности**: Этот документ представляет собой компиляцию информации, собранной из интернета, и не гарантирует функциональность во всех средах. Описанные процедуры могут работать некорректно из-за различий в окружениях и версиях. Пожалуйста, адаптируйте их к вашей конкретной среде по мере необходимости. Также рекомендуется всегда обращаться к актуальной информации в официальном репозитории.

FramePack-eichi — это система генерации видео на основе ИИ, которая создает короткие видеоролики из одного изображения с использованием текстовых подсказок. Это форк оригинального FramePack, разработанного Львмином Чжаном и Манишем Агравалой из Стэнфордского университета, с многочисленными дополнительными функциями и улучшениями. Это руководство предоставляет точные процедуры настройки для каждой среды, системные требования и советы по устранению неполадок.

## Системные требования

### Требования к оперативной памяти
- **Минимум**: 16 ГБ (будет работать, но с ограничениями производительности)
- **Рекомендуется**: 32 ГБ (достаточно для стандартных операций)
- **Оптимально**: 64 ГБ (идеально для длинных видео, использования LoRA и обработки с высоким разрешением)
- Если доступной оперативной памяти недостаточно, система будет использовать пространство подкачки на SSD, что может сократить срок службы вашего SSD

### Требования к видеопамяти (VRAM)
- **Минимум**: 8 ГБ VRAM (рекомендуемый минимум для FramePack-eichi)
- **Режим низкой VRAM**: Автоматически активируется и эффективно управляет памятью
  - Регулируется через настройку `gpu_memory_preservation` (по умолчанию: 10 ГБ)
  - Меньшее значение = Больше VRAM для обработки = Быстрее, но рискованнее
  - Большее значение = Меньше VRAM для обработки = Медленнее, но стабильнее
- **Режим высокой VRAM**: Автоматически активируется при обнаружении более 100 ГБ свободной VRAM
  - Модели остаются в памяти GPU (примерно на 20% быстрее)
  - Нет необходимости в периодической загрузке/выгрузке моделей

### Требования к процессору
- Конкретная минимальная модель процессора не указана
- **Рекомендуется**: Современный многоядерный процессор с 8+ ядрами
- Производительность процессора влияет на время загрузки и предпроцессинг/постпроцессинг
- Большая часть фактической генерации выполняется на GPU

### Требования к хранилищу
- **Код приложения**: Обычно 1-2 ГБ
- **Модели**: Около 30 ГБ (автоматически загружаются при первом запуске)
- **Выходные и временные файлы**: Зависит от длины видео, разрешения и настроек сжатия
- **Общая рекомендуемая ёмкость**: 150 ГБ или больше
- Рекомендуется SSD для частых операций чтения/записи

### Поддерживаемые модели GPU
- **Официально поддерживаемые**: Серии NVIDIA RTX 30XX, 40XX, 50XX (поддерживающие форматы данных fp16 и bf16)
- **Минимально рекомендуемые**: RTX 3060 (или эквивалент с 8 ГБ+ VRAM)
- **Подтверждено работающие**: RTX 3060, 3070Ti, 4060Ti, 4090
- **Неофициально/Не тестировано**: Серии GTX 10XX/20XX
- **AMD GPU**: Явная поддержка не указана
- **Intel GPU**: Явная поддержка не указана

## Инструкции по настройке для Windows

### Предварительные требования
- Windows 10/11
- NVIDIA GPU с драйверами, поддерживающими CUDA 12.6
- Python 3.10.x
- 7-Zip (для распаковки установочных пакетов)

### Пошаговые инструкции
1. **Установка базового FramePack**:
   - Перейдите в [официальный репозиторий FramePack](https://github.com/lllyasviel/FramePack)
   - Нажмите "Download One-Click Package (CUDA 12.6 + PyTorch 2.6)"
   - Скачайте и распакуйте 7z-архив в место по вашему выбору
   - Запустите `update.bat` (важно для получения последних исправлений ошибок)
   - Запустите `run.bat`, чтобы запустить FramePack в первый раз
   - Необходимые модели (около 30 ГБ) будут автоматически загружены при первом запуске

2. **Установка FramePack-eichi**:
   - Клонируйте или скачайте [репозиторий FramePack-eichi](https://github.com/git-ai-code/FramePack-eichi)
   - Скопируйте соответствующий языковой пакетный файл (`run_endframe_ichi.bat` для японского, `run_endframe_ichi_en.bat` для английского, `run_endframe_ichi_zh-tw.bat` для традиционного китайского, `run_endframe_ichi_ru.bat` для русского) в корневую директорию FramePack
   - Для версии F1 используйте `run_endframe_ichi_f1.bat` (или языковую версию `run_endframe_ichi_en_f1.bat`, `run_endframe_ichi_zh-tw_f1.bat`, `run_endframe_ichi_f1_ru.bat`)
   - Для однокадрового вывода используйте `run_oneframe_ichi.bat` (или языковую версию `run_oneframe_ichi_en.bat`, `run_oneframe_ichi_zh-tw.bat`, `run_oneframe_ichi_ru.bat`)
   - Скопируйте следующие файлы/папки из FramePack-eichi в папку `webui` в FramePack:
     - `endframe_ichi.py`
     - папка `eichi_utils` (включает `lora_preset_manager.py`, `model_downloader.py`, `vae_settings.py` и др. - добавлено в v1.9.3)
     - папка `lora_utils`
     - папка `diffusers_helper`
     - папка `locales` (включает `ru.json` файл русского перевода - добавлено в v1.9.3)

3. **Установка библиотек ускорения (Опционально, но рекомендуется)**:
   - Скачайте установщик ускоряющих пакетов из [FramePack Issue #138](https://github.com/lllyasviel/FramePack/issues/138)
   - Распакуйте файл `package_installer.zip` в корневую директорию FramePack
   - Запустите `package_installer.bat` и следуйте инструкциям на экране (обычно просто нажимайте Enter)
   - Перезапустите FramePack и убедитесь, что в консоли отображаются следующие сообщения:
     ```
     Xformers is installed!
     Flash Attn is not installed! (This is normal)
     Sage Attn is installed!
     ```

4. **Запуск FramePack-eichi**:
   - Запустите `run_endframe_ichi.bat` (или соответствующую языковую версию) из корневой директории FramePack
   - WebUI откроется в вашем браузере по умолчанию

5. **Проверка**:
   - Загрузите изображение в WebUI
   - Введите подсказку, описывающую желаемое движение
   - Нажмите "Начать генерацию", чтобы убедиться, что генерация видео работает

## Инструкции по настройке для Linux

### Поддерживаемые дистрибутивы Linux
- Ubuntu 22.04 LTS и новее (основная поддержка)
- Другие дистрибутивы, поддерживающие Python 3.10, также должны работать

### Необходимые пакеты и зависимости
- Драйверы NVIDIA GPU, поддерживающие CUDA 12.6
- Python 3.10.x
- CUDA Toolkit 12.6
- PyTorch 2.6 с поддержкой CUDA

### Шаги установки

1. **Настройка окружения Python**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Установка PyTorch с поддержкой CUDA**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

3. **Клонирование и настройка FramePack**:
   ```bash
   git clone https://github.com/lllyasviel/FramePack.git
   cd FramePack
   pip install -r requirements.txt
   ```

4. **Клонирование и настройка FramePack-eichi**:
   ```bash
   git clone https://github.com/git-ai-code/FramePack-eichi.git
   # Копирование необходимых файлов
   cp FramePack-eichi/webui/endframe_ichi.py FramePack/
   cp FramePack-eichi/webui/endframe_ichi_f1.py FramePack/
   cp -r FramePack-eichi/webui/eichi_utils FramePack/
   cp -r FramePack-eichi/webui/lora_utils FramePack/
   cp -r FramePack-eichi/webui/diffusers_helper FramePack/
   cp -r FramePack-eichi/webui/locales FramePack/
   ```

5. **Установка библиотек ускорения (Опционально)**:
   ```bash
   # sage-attention (рекомендуется)
   pip install sageattention==1.0.6
   
   # xformers (если поддерживается)
   pip install xformers
   ```

6. **Запуск FramePack-eichi**:
   ```bash
   cd FramePack
   python endframe_ichi.py  # По умолчанию японский интерфейс
   python endframe_ichi_f1.py  # По умолчанию японский интерфейс
   # Для английского интерфейса:
   python endframe_ichi.py --lang en
   python endframe_ichi_f1.py --lang en
   # Для традиционного китайского интерфейса:
   python endframe_ichi.py --lang zh-tw
   python endframe_ichi_f1.py --lang zh-tw
   ```

## Инструкции по настройке Docker

### Предварительные требования
- Docker установлен в вашей системе
- Docker Compose установлен
- NVIDIA Container Toolkit установлен для использования GPU

### Процесс настройки Docker

1. **Использование Docker-реализации akitaonrails**:
   ```bash
   git clone https://github.com/akitaonrails/FramePack-Docker-CUDA.git
   cd FramePack-Docker-CUDA
   mkdir outputs
   mkdir hf_download
   
   # Создание Docker-образа
   docker build -t framepack-torch26-cu124:latest .
   
   # Запуск контейнера с поддержкой GPU
   docker run -it --rm --gpus all -p 7860:7860 \
   -v ./outputs:/app/outputs \
   -v ./hf_download:/app/hf_download \
   framepack-torch26-cu124:latest
   ```

2. **Альтернативная настройка Docker Compose**:
   - Создайте файл `docker-compose.yml` со следующим содержимым:
   ```yaml
   version: '3'
   services:
     framepack:
       build: .
       ports:
         - "7860:7860"
       volumes:
         - ./outputs:/app/outputs
         - ./hf_download:/app/hf_download
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: all
                 capabilities: [gpu]
       # Выбор языка (по умолчанию английский)
       command: ["--lang", "en"]  # Опции: "ja" (японский), "zh-tw" (традиционный китайский), "en" (английский)
   ```
   
   - Создайте `Dockerfile` в той же директории:
   ```dockerfile
   FROM python:3.10-slim
   
   ENV DEBIAN_FRONTEND=noninteractive
   
   # Установка системных зависимостей
   RUN apt-get update && apt-get install -y \
       git \
       wget \
       ffmpeg \
       && rm -rf /var/lib/apt/lists/*
   
   # Настройка рабочей директории
   WORKDIR /app
   
   # Клонирование репозиториев
   RUN git clone https://github.com/lllyasviel/FramePack.git . && \
       git clone https://github.com/git-ai-code/FramePack-eichi.git /tmp/FramePack-eichi
   
   # Копирование файлов FramePack-eichi (в корневую директорию, так же как в настройке Linux)
   RUN cp /tmp/FramePack-eichi/webui/endframe_ichi.py . && \
       cp /tmp/FramePack-eichi/webui/endframe_ichi_f1.py . && \
       cp -r /tmp/FramePack-eichi/webui/eichi_utils . && \
       cp -r /tmp/FramePack-eichi/webui/lora_utils . && \
       cp -r /tmp/FramePack-eichi/webui/diffusers_helper . && \
       cp -r /tmp/FramePack-eichi/webui/locales . && \
       rm -rf /tmp/FramePack-eichi
   
   # Установка PyTorch и зависимостей
   RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   RUN pip install -r requirements.txt
   RUN pip install sageattention==1.0.6
   
   # Создание выходных директорий
   RUN mkdir -p outputs hf_download
   
   # Настройка кэш-директории HuggingFace
   ENV HF_HOME=/app/hf_download
   
   # Открытие порта для WebUI
   EXPOSE 7860
   
   # Запуск FramePack-eichi (из корневой директории, так же как в настройке Linux)
   ENTRYPOINT ["python", "endframe_ichi.py", "--listen"]
   ```
   
   - Создание и запуск с Docker Compose:
   ```bash
   docker-compose build
   docker-compose up
   ```

3. **Доступ к WebUI**:
   - После запуска контейнера WebUI будет доступен по адресу http://localhost:7860

4. **Конфигурация передачи GPU**:
   - Убедитесь, что NVIDIA Container Toolkit установлен правильно
   - Параметр `--gpus all` (или его эквивалент в docker-compose.yml) необходим для передачи GPU
   - Проверьте, доступны ли GPU внутри контейнера:
     ```bash
     docker exec -it [container_id] nvidia-smi
     ```

## Инструкции по настройке для macOS (Apple Silicon)

FramePack-eichi может использоваться на Mac с Apple Silicon через форк brandon929/FramePack, который использует Metal Performance Shaders вместо CUDA.

### Предварительные требования
- macOS с Apple Silicon (чип M1, M2 или M3)
- Homebrew (пакетный менеджер macOS)
- Python 3.10
- **Требования к памяти**: Минимум 16 ГБ RAM, рекомендуется 32 ГБ+
  - Модели с 8 ГБ, вероятно, будут испытывать серьезное ухудшение производительности и ошибки обработки
  - Модели с 16 ГБ будут ограничены короткими видео (3-5 секунд) и настройками низкого разрешения
  - Модели с 32 ГБ+ позволяют комфортную обработку (рекомендуется M2/M3 Ultra)

### Шаги установки

1. **Установка Homebrew** (если еще не установлен):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   - Следуйте дополнительным инструкциям, чтобы добавить Homebrew в ваш PATH.

2. **Установка Python 3.10**:
   ```bash
   brew install python@3.10
   ```

3. **Клонирование форка, совместимого с macOS**:
   ```bash
   git clone https://github.com/brandon929/FramePack.git
   cd FramePack
   ```

4. **Установка PyTorch с поддержкой Metal** (CPU-версия, поддержка Metal добавляется через PyTorch MPS):
   ```bash
   pip3.10 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
   ```

5. **Установка зависимостей**:
   ```bash
   pip3.10 install -r requirements.txt
   ```

6. **Запуск веб-интерфейса**:
   ```bash
   python3.10 demo_gradio.py --fp32
   ```
   
   Флаг `--fp32` важен для совместимости с Apple Silicon. Процессоры M1/M2/M3 могут не полностью поддерживать float16 и bfloat16, используемые в оригинальных моделях.

7. **После запуска** откройте веб-браузер и перейдите по URL, отображаемому в терминале (обычно http://127.0.0.1:7860).

### Особые соображения для Apple Silicon

- **Производительность Metal**: 
  - Используйте флаг `--fp32` для совместимости с Apple Silicon
- **Настройки разрешения**: 
  - 16 ГБ RAM: Рекомендуется максимальное разрешение 416×416
  - 32 ГБ RAM: Рекомендуется максимальное разрешение 512×512
  - 64 ГБ RAM: Можно попробовать максимальное разрешение 640×640
- **Сравнение производительности**:
  - Скорость генерации значительно медленнее по сравнению с GPU NVIDIA
  - Сравнение времени генерации 5-секундного видео:
    - RTX 4090: ~6 минут
    - M2 Max: ~25-30 минут
    - M3 Max: ~20-25 минут
    - M2 Ultra: ~15-20 минут
    - M3 Ultra: ~12-15 минут
- **Управление памятью**: 
  - Унифицированная архитектура памяти Apple Silicon означает, что GPU/CPU используют один и тот же пул памяти
  - Следите за "Memory Pressure" в Activity Monitor и уменьшайте настройки, если компрессия высокая
  - Увеличенное использование подкачки резко снизит производительность и повлияет на срок службы SSD
  - Настоятельно рекомендуется закрыть другие ресурсоемкие приложения во время генерации
  - Перезапускайте приложение после длительного использования для устранения утечек памяти

## Инструкции по настройке WSL

Настройка FramePack-eichi в WSL обеспечивает Linux-среду в Windows с ускорением GPU через драйверы NVIDIA для WSL.

### Предварительные требования
- Windows 10 (версия 2004 или новее) или Windows 11
- NVIDIA GPU (рекомендуется серия RTX 30XX, 40XX или 50XX, минимум 8 ГБ VRAM)
- Доступ администратора
- Обновленные драйверы NVIDIA с поддержкой WSL2

### Шаги установки

1. **Установка WSL2**:
   
   Откройте PowerShell от имени администратора и выполните:
   ```powershell
   wsl --install
   ```
   
   Эта команда устанавливает WSL2 с Ubuntu в качестве дистрибутива Linux по умолчанию. Перезагрузите компьютер при появлении запроса.

2. **Проверка правильной установки WSL2**:
   ```powershell
   wsl --status
   ```
   
   Убедитесь, что "WSL 2" указан как версия по умолчанию.

3. **Обновление ядра WSL** (при необходимости):
   ```powershell
   wsl --update
   ```

4. **Установка драйверов NVIDIA для WSL**:
   
   Скачайте и установите последние драйверы NVIDIA с поддержкой WSL с сайта NVIDIA. Не устанавливайте драйверы NVIDIA внутри среды WSL — WSL использует драйверы Windows.

5. **Запустите Ubuntu и проверьте доступ к GPU**:
   
   Запустите Ubuntu из меню Пуск или выполните `wsl` в PowerShell и проверьте обнаружение NVIDIA GPU:
   ```bash
   nvidia-smi
   ```
   
   Вы должны увидеть отображаемую информацию о вашем GPU.

6. **Настройка окружения в WSL**:
   ```bash
   # Обновление списков пакетов
   sudo apt update && sudo apt upgrade -y
   
   # Установка Python и инструментов разработки
   sudo apt install -y python3.10 python3.10-venv python3-pip git
   
   # Клонирование репозитория FramePack-eichi
   git clone https://github.com/git-ai-code/FramePack-eichi.git
   cd FramePack-eichi
   
   # Создание и активация виртуального окружения
   python3.10 -m venv venv
   source venv/bin/activate
   
   # Установка PyTorch с поддержкой CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   
   # Установка зависимостей
   pip install -r requirements.txt
   ```

7. **Запуск FramePack-eichi**:
   ```bash
   python endframe_ichi.py
   ```

   Вы также можете указать язык:
   ```bash
   python endframe_ichi.py --lang en  # Для английского
   ```

8. **Доступ к веб-интерфейсу** через открытие браузера в Windows и переход по URL, отображаемому в терминале (обычно http://127.0.0.1:7860).

## Инструкции по настройке окружения Anaconda

### Создание нового окружения Conda

```bash
# Создание нового окружения conda с Python 3.10
conda create -n framepack-eichi python=3.10
conda activate framepack-eichi

# Установка PyTorch с поддержкой CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Ручная установка из исходников

```bash
# Клонирование оригинального репозитория FramePack
git clone https://github.com/lllyasviel/FramePack.git
cd FramePack

# Клонирование репозитория FramePack-eichi во временное место
git clone https://github.com/git-ai-code/FramePack-eichi.git temp_eichi

# Копирование расширенных файлов webui (в корневую директорию, так же как в настройке Linux)
cp temp_eichi/webui/endframe_ichi.py .
cp temp_eichi/webui/endframe_ichi_f1.py .
cp -r temp_eichi/webui/eichi_utils .
cp -r temp_eichi/webui/lora_utils .
cp -r temp_eichi/webui/diffusers_helper .
cp -r temp_eichi/webui/locales .

# Копирование специфичных для языка пакетных файлов в корневую директорию (выберите соответствующий файл)
cp temp_eichi/run_endframe_ichi.bat .  # Японский (по умолчанию)
# cp temp_eichi/run_endframe_ichi_en.bat .  # Английский
# cp temp_eichi/run_endframe_ichi_zh-tw.bat .  # Традиционный китайский

# Установка зависимостей
pip install -r requirements.txt

# Очистка временной директории
rm -rf temp_eichi
```

### Особые соображения для Conda

- При установке через conda могут возникнуть конфликты зависимостей с пакетами PyTorch
- Для достижения наилучших результатов устанавливайте PyTorch, torchvision и torchaudio через pip, используя официальный URL индекса, а не каналы conda
- Опциональные пакеты ускорения, такие как xformers, flash-attn и sageattention, следует устанавливать отдельно после создания основного окружения

## Инструкции по настройке Google Colab

### Последняя настройка Colab на май 2025 года (наиболее стабильная)

Следующий скрипт обеспечивает наиболее стабильную настройку для последней среды Colab (по состоянию на май 2025 года). Он был специально протестирован в среде GPU A100.

```python
# Установка git, если он еще не установлен
!apt-get update && apt-get install -y git

# Клонирование репозитория FramePack
!git clone https://github.com/lllyasviel/FramePack.git
%cd FramePack

# Установка PyTorch (версия с поддержкой CUDA)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Обновление Requests и NumPy для среды Colab
!pip install requests==2.32.3 numpy==2.0.0

# Установка зависимостей FramePack
!pip install -r requirements.txt

# Установка SageAttention для оптимизации скорости (опционально)
!pip install sageattention==1.0.6

# Запуск демо FramePack (раскомментируйте для запуска)
# !python demo_gradio.py --share

# Установка FramePack-eichi
!git clone https://github.com/git-ai-code/FramePack-eichi.git tmp
!rsync -av --exclude='diffusers_helper' tmp/webui/ ./
!cp tmp/webui/diffusers_helper/bucket_tools.py diffusers_helper/
!cp tmp/webui/diffusers_helper/memory.py diffusers_helper/
!rm -rf tmp

# Запуск FramePack-eichi
!python endframe_ichi.py --share
```

> **Важно**: Вышеуказанный метод специально копирует файл `diffusers_helper/bucket_tools.py` индивидуально. Это необходимо для избежания распространенной ошибки "ImportError: cannot import name 'SAFE_RESOLUTIONS' from 'diffusers_helper.bucket_tools'".

### Альтернативный метод настройки Colab

Ниже приведен альтернативный метод настройки. Предпочтительно использовать вышеуказанный метод для более новых сред.

```python
# Клонирование репозитория FramePack-eichi
!git clone https://github.com/git-ai-code/FramePack-eichi.git tmp

# Клонирование базового FramePack
!git clone https://github.com/lllyasviel/FramePack.git
%cd /content/FramePack

# Установка зависимостей
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
!pip install -r requirements.txt

# Настройка расширений eichi (в корневую директорию, так же как в настройке Linux)
!cp /content/tmp/webui/endframe_ichi.py .
!cp /content/tmp/webui/endframe_ichi_f1.py .
!cp -r /content/tmp/webui/eichi_utils .
!cp -r /content/tmp/webui/lora_utils .
!cp -r /content/tmp/webui/diffusers_helper .
!cp -r /content/tmp/webui/locales .
!cp /content/tmp/run_endframe_ichi.bat .

# Установка переменной окружения PYTHONPATH
%env PYTHONPATH=/content/FramePack:$PYTHONPATH

# Запуск WebUI с публичным URL
%cd /content/FramePack
!python endframe_ichi.py --share
```

### Интеграция с Google Drive и конфигурация вывода

Для сохранения сгенерированных видео в Google Drive:

```python
# Монтирование Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Настройка выходной директории
import os
OUTPUT_DIR = "/content/drive/MyDrive/FramePack-eichi-outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Запуск framepack с указанной выходной директорией
!python endframe_ichi.py --share --output_dir={OUTPUT_DIR}
```

### Общее устранение неполадок для Colab

1. **Ошибка импорта 'SAFE_RESOLUTIONS'**:
   ```
   ImportError: cannot import name 'SAFE_RESOLUTIONS' from 'diffusers_helper.bucket_tools'
   ```
   - **Решение**: Используйте последний скрипт настройки от мая 2025 года, который включает индивидуальное копирование файлов diffusers_helper

2. **Ошибки нехватки памяти**:
   ```
   RuntimeError: CUDA out of memory
   ```
   - **Решения**: 
     - Уменьшите разрешение (например, 416×416)
     - Уменьшите количество ключевых кадров
     - Уменьшите размер пакета
     - Настройте параметр сохранения памяти GPU

3. **Отключение сессии**:
   - **Решения**:
     - Избегайте длительного времени обработки
     - Сохраняйте прогресс в Google Drive
     - Поддерживайте активность вкладки браузера

### Соображения по VRAM/RAM для различных уровней Colab

| Уровень Colab | Тип GPU | VRAM | Производительность | Примечания |
|------------|----------|------|-------------|-------|
| Free       | T4       | 16GB | Ограниченная | Достаточно для базового использования с короткими видео (1-5 секунд) |
| Pro        | A100     | 40GB | Хорошая      | Может обрабатывать более длинные видео и несколько ключевых кадров |
| Pro+       | A100     | 80GB | Отличная     | Лучшая производительность, способна к сложным генерациям |

### Оптимальные настройки для Colab

1. **Настройки аппаратного ускорителя**:
   - Меню "Runtime" → "Change runtime type" → Установите "Hardware accelerator" на "GPU"
   - Пользователи Pro/Pro+ должны выбрать опцию "High RAM" или "High-memory", если доступно

2. **Рекомендуемые настройки размера пакета и разрешения**:
   - **GPU T4 (Free)**: Размер пакета 4, разрешение 416x416
   - **GPU A100 (Pro)**: Размер пакета 8, разрешение до 640x640
   - **GPU A100 (Pro+/High-memory)**: Размер пакета 16, разрешение до 768x768

## Инструкции по настройке облачной среды (AWS/GCP/Azure)

### Настройка AWS EC2

#### Рекомендуемые типы инстансов:
- **g4dn.xlarge**: 1 NVIDIA T4 GPU (16GB), 4 vCPU, 16GB RAM
- **g4dn.2xlarge**: 1 NVIDIA T4 GPU (16GB), 8 vCPU, 32GB RAM
- **g5.xlarge**: 1 NVIDIA A10G GPU (24GB), 4 vCPU, 16GB RAM
- **p3.2xlarge**: 1 NVIDIA V100 GPU (16GB), 8 vCPU, 61GB RAM

#### Шаги настройки:

1. **Запуск EC2 инстанса** - Используйте Deep Learning AMI (Ubuntu) с выбранным типом инстанса
2. **Подключение к инстансу через SSH**:
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```
3. **Обновление системных пакетов**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
4. **Клонирование репозиториев**:
   ```bash
   git clone https://github.com/lllyasviel/FramePack.git
   cd FramePack
   git clone https://github.com/git-ai-code/FramePack-eichi.git temp_eichi
   # Копирование файлов в корневую директорию, так же как в настройке Linux
   cp temp_eichi/webui/endframe_ichi.py .
   cp temp_eichi/webui/endframe_ichi_f1.py .
   cp -r temp_eichi/webui/eichi_utils .
   cp -r temp_eichi/webui/lora_utils .
   cp -r temp_eichi/webui/diffusers_helper .
   cp -r temp_eichi/webui/locales .
   cp temp_eichi/run_endframe_ichi_en.bat .  # Английская версия
   rm -rf temp_eichi
   ```
5. **Установка зависимостей**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```
6. **Настройка группы безопасности** - Разрешите входящий трафик на порт 7860
7. **Запуск с публичной видимостью**:
   ```bash
   python endframe_ichi.py --server --listen --port 7860
   ```
8. **Доступ к UI** - http://your-instance-ip:7860

### Настройка Google Cloud Platform (GCP)

#### Рекомендуемые типы инстансов:
- **n1-standard-8** + 1 NVIDIA T4 GPU
- **n1-standard-8** + 1 NVIDIA V100 GPU
- **n1-standard-8** + 1 NVIDIA A100 GPU

#### Шаги настройки:

1. **Создание VM-инстанса** с образом Deep Learning VM
2. **Включение GPU** и выбор соответствующего типа GPU
3. **Подключение к инстансу через SSH**
4. **Следуйте тем же шагам, что и для AWS EC2** для настройки FramePack-eichi
5. **Настройка правил брандмауэра** - Разрешите входящий трафик на порт 7860

### Настройка Microsoft Azure

#### Рекомендуемые размеры VM:
- **Standard_NC6s_v3**: 1 NVIDIA V100 GPU (16GB)
- **Standard_NC4as_T4_v3**: 1 NVIDIA T4 GPU (16GB)
- **Standard_NC24ads_A100_v4**: 1 NVIDIA A100 GPU (80GB)

#### Шаги настройки:
1. **Создание VM** с Data Science Virtual Machine (Ubuntu)
2. **Подключение к VM через SSH**
3. **Следуйте тем же шагам, что и для AWS EC2** для настройки FramePack-eichi
4. **Настройка группы сетевой безопасности** - Разрешите входящий трафик на порт 7860

## Распространенные ошибки и процедуры устранения неполадок

### Ошибки установки

#### Конфликты зависимостей Python
- **Симптомы**: Сообщения об ошибках о несовместимых версиях пакетов
- **Решения**: 
  - Явно используйте Python 3.10 (не 3.11, 3.12 или выше)
  - Установите PyTorch с правильной версией CUDA
  - Создайте новое виртуальное окружение, если возникают ошибки зависимостей

#### Проблемы с установкой и совместимостью CUDA
- **Симптомы**: Ошибки "CUDA is not available", предупреждения о запуске на CPU
- **Решения**:
  - Убедитесь, что вы используете поддерживаемый GPU (рекомендуется серия RTX 30XX, 40XX или 50XX)
  - Установите правильный CUDA toolkit (рекомендуется 12.6)
  - Проведите отладку в Python:
    ```python
    import torch
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    ```

#### Сбои установки пакетов
- **Симптомы**: Ошибки установки PIP, сбои сборки wheel
- **Решения**:
  - Используйте установщик в один клик для Windows (вместо ручной установки)
  - Для Linux: Установите необходимые зависимости сборки
  - Если установка SageAttention не удается, вы можете продолжить без него
  - Используйте package_installer.zip из Issue #138 для установки пакетов расширенной оптимизации

### Ошибки времени выполнения

#### Ошибки нехватки памяти CUDA
- **Симптомы**: Сообщения об ошибках "CUDA out of memory", сбои во время фаз генерации с высоким потреблением памяти
- **Решения**:
  - Увеличьте значение `gpu_memory_preservation` (попробуйте значения между 6-16 ГБ)
  - Закройте другие приложения, интенсивно использующие GPU
  - Перезапустите и попробуйте снова
  - Уменьшите разрешение изображения (рекомендуется 512x512 для низкой VRAM)
  - Установите больший файл подкачки Windows (3x физической RAM)
  - Обеспечьте достаточную системную RAM (рекомендуется 32 ГБ+)

#### Сбои загрузки модели
- **Симптомы**: Сообщения об ошибках при загрузке фрагментов модели, сбои процесса при инициализации модели
- **Решения**:
  - Запустите `update.bat` перед запуском приложения
  - Убедитесь, что все модели правильно загружены в папку `webui/hf_download`
  - Если модели отсутствуют, разрешите автоматической загрузке завершиться (около 30 ГБ)
  - Если вы вручную размещаете модели, копируйте файлы в правильную папку `framepack\webui\hf_download`

#### Проблемы с запуском WebUI
- **Симптомы**: Интерфейс Gradio не появляется после запуска, браузер показывает ошибку "не удается подключиться"
- **Решения**:
  - Попробуйте другой порт с опцией командной строки `--port XXXX`
  - Убедитесь, что нет других приложений, использующих порт 7860 (по умолчанию для Gradio)
  - Используйте опцию `--inbrowser` для автоматического открытия браузера
  - Проверьте консольный вывод на наличие конкретных сообщений об ошибках

### Проблемы, специфичные для платформы

#### Проблемы, специфичные для Windows
- **Симптомы**: Ошибки, связанные с путями, сбои загрузки DLL, пакетные файлы не выполняются должным образом
- **Решения**:
  - Установите в короткий путь (например, C:\FramePack), чтобы избежать ограничений длины пути
  - Запускайте пакетные файлы от имени администратора, если возникают проблемы с разрешениями
  - Если появляются ошибки загрузки DLL:
    - Установите пакеты Visual C++ Redistributable
    - Проверьте, не блокирует ли антивирусное ПО выполнение

#### Проблемы, специфичные для Linux
- **Симптомы**: Ошибки отсутствующих библиотек, сбои сборки пакетов, проблемы с отображением GUI
- **Решения**:
  - В Debian/Ubuntu установите необходимые системные библиотеки:
    ```
    sudo apt-get install libavformat-dev libavdevice-dev libavfilter-dev libswscale-dev libopenblas-dev
    ```
  - При проблемах с обнаружением GPU убедитесь, что драйверы NVIDIA установлены корректно:
    ```
    nvidia-smi
    ```

#### Проблемы, специфичные для macOS
- **Симптомы**: Ошибки, связанные с Metal/MPS, низкая производительность на Apple Silicon
- **Решения**:
  - Запускайте с флагом `--fp32` (M1/M2 могут не полностью поддерживать fp16/bf16)
  - При проблемах с форматом видео настройте параметры сжатия MP4 примерно на 16 (по умолчанию)
  - Учитывайте значительно сниженную производительность по сравнению с оборудованием NVIDIA

#### Проблемы настройки WSL
- **Симптомы**: GPU не обнаруживается в WSL, крайне низкая производительность в WSL
- **Решения**:
  - Убедитесь, что вы используете WSL2 (не WSL1): `wsl --set-version <Distro> 2`
  - Установите специальные драйверы NVIDIA для WSL
  - Создайте файл `.wslconfig` в директории пользователя Windows:
    ```
    [wsl2]
    memory=16GB  # Настройте в соответствии с вашей системой
    processors=8  # Настройте в соответствии с вашей системой
    gpumemory=8GB  # Настройте в соответствии с вашим GPU
    ```

### Проблемы с производительностью

#### Медленное время генерации и техники оптимизации
- **Симптомы**: Чрезмерно долгое время генерации, производительность ниже ожидаемой по сравнению с эталонными тестами
- **Решения**:
  - Установите библиотеки оптимизации:
    - Скачайте package_installer.zip из Issue #138 и запустите package_installer.bat
    - Это установит xformers, flash-attn и sage-attn, где возможно
  - Включите teacache для более быстрой (но потенциально менее качественной) генерации
  - Закройте другие приложения, интенсивно использующие GPU
  - Уменьшите разрешение для более быстрой генерации (за счет качества)

#### Утечки памяти и управление памятью
- **Симптомы**: Увеличение использования памяти со временем, ухудшение производительности в течение нескольких генераций
- **Решения**:
  - Перезапускайте приложение между длительными сессиями генерации
  - Отслеживайте использование памяти GPU:
    ```
    nvidia-smi -l 1
    ```
  - Перезапускайте процесс Python, если возникают утечки CPU/памяти
  - Используйте явную выгрузку модели при переключении настроек
  - Не загружайте несколько LoRA одновременно, если это не требуется

## Источники информации

1. Официальные репозитории:
   - FramePack-eichi: https://github.com/git-ai-code/FramePack-eichi
   - Оригинальный FramePack: https://github.com/lllyasviel/FramePack

2. Ресурсы сообщества:
   - Docker-реализация FramePack: https://github.com/akitaonrails/FramePack-Docker-CUDA
   - Форк, совместимый с Apple Silicon: https://github.com/brandon929/FramePack

3. Официальная документация:
   - README и wiki репозитория FramePack-eichi на GitHub
   - Комментарии разработчиков в GitHub Issues

4. Ресурсы по устранению неполадок:
   - FramePack Issue #138 (Библиотеки ускорения): https://github.com/lllyasviel/FramePack/issues/138
   - Документация по конфигурации WSL CUDA: https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl

Это руководство предоставляет исчерпывающие инструкции по настройке FramePack-eichi и лучшие практики для работы в различных средах. Выберите оптимальный путь настройки для вашей среды и обращайтесь к процедурам устранения неполадок по мере необходимости.