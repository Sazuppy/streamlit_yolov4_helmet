import cv2 as cv
import streamlit as st
import tempfile
import time
from moviepy.editor import VideoClip
from moviepy.editor import AudioFileClip
import datetime
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
from moviepy.editor import concatenate_videoclips
import os
@st.cache_resource
def yolov4(names, weights, config, Conf_threshold, NMS_threshold):
    Conf_threshold = Conf_threshold
    NMS_threshold = NMS_threshold
    COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    class_name = []
    with open(names, 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]
    # Инициализация нейросети YOLOv4 
    net = cv.dnn.readNet(weights, config)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
    return model, COLORS, class_name

def split_video(input_file, duration=15):
    output_folder = "temp_video"
    # Получение длительности входного видео
    video_clip = VideoFileClip(input_file)
    total_duration = video_clip.duration

    # Разбивка видео на куски по 15 секунд
    start_time = 0
    end_time = duration
    clip_number = 1
    created_files = []
    
    while end_time <= total_duration:
        output_file = f"{output_folder}/clip_{clip_number}.{input_file.split('.')[-1]}"
        subclip = video_clip.subclip(start_time, end_time)
        subclip.write_videofile(output_file, codec="libx264", audio_codec="aac", fps=24)
        created_files.append(output_file)
        start_time += duration
        end_time += duration
        clip_number += 1

    # Если последний кусок короче 15 секунд, создаем дополнительный кусок
    if start_time < total_duration:
        output_file = f"{output_folder}/clip_{clip_number}.{input_file.split('.')[-1]}"
        subclip = video_clip.subclip(start_time, total_duration)
        subclip.write_videofile(output_file, codec="libx264", audio_codec="aac", fps=24)
        created_files.append(output_file)
    return created_files

def temp_file_image(data):    
    # Сохранение изображения во временный файл
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{data}')
    temp_file.write(data.read())
    temp_file_path = temp_file.name
    cap = cv.VideoCapture(data)
    temp_file.close() 
    return cap, temp_file_path

def temp_file(data, video_filename):    
    # Сохранение видеофайла во временный файл
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{video_filename}')
    temp_file.write(data.read())
    temp_file_path = temp_file.name
    temp_file.close() 
    return temp_file_path

def temp_file_for_process(data):
    # Захват видео
    cap = cv.VideoCapture(data)
    
    return cap, data

def image_process(model, cap, Conf_threshold, NMS_threshold, COLORS, class_name):
    
    ret, frame = cap.read()
    # Обнаружение объектов с использованием YOLOv4
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid], score)
        cv.rectangle(frame, box, color, 1)
        cv.putText(frame, label, (box[0], box[1]-10),
                cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
    frame = cv.resize(frame,(0,0), fx=0.8, fy=0.8)
    return frame       



def video_process(model, cap, Conf_threshold, NMS_threshold, COLORS, class_name, input_file):
    starting_time = time.time()
    frame_counter = 0
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    remaining_time = datetime.timedelta(seconds=0)
    remaining_time_container = st.empty()
    video_frames = []
    num_fiiles = len(created_files)
    num = input_file.split('.')[0][-1]
    while True:
        ret, frame = cap.read()
        frame_counter += 1
        if ret == False:
            break
        # Обнаружение объектов с использованием YOLOv4
        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_name[classid], score)
            cv.rectangle(frame, box, color, 1)
            cv.putText(frame, label, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
        if app_mode == 'Видео':
            endingTime = time.time() - starting_time
            fps = frame_counter/endingTime
            remaining_frames = total_frames - frame_counter
            remaining_time = datetime.timedelta(seconds=int(remaining_frames / fps))
            
        # Преобразование кадра в формат RGB для отображения в Streamlit
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        video_frames.append(frame.copy())
        remaining_time_container.markdown(f'Оставшееся время выполнения: {str(remaining_time)}, {num} из {num_fiiles}')           
        
        if stop:
            break
       
    return video_frames

def concotinate_video(video_frames, data, output_file_path):
    cap = cv.VideoCapture(data)
    # Считывание видео для получения FPS
    fps_original = cap.get(cv.CAP_PROP_FPS)
    total_frames_original = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    # Создание функции для извлечения кадров из списка video_frames
    def get_frame(t):
        index = int(t * fps_original)
        if index < total_frames_original:
            return video_frames[index]
        else:
            return video_frames[-1]
    # Считывание аудио для получения длительности
    audio_clip = AudioFileClip(data)
    
    video_clip = VideoClip(get_frame, duration=total_frames_original / fps_original)
    video_clip = video_clip.set_audio(audio_clip)

    video_clip.write_videofile(output_file_path, codec="libx264", audio_codec="aac", fps=fps_original)
    
def concatenate_videos_in_one(file_paths):
    st.markdown('Сборка видео, ожидайте...')
    # Создание списка VideoFileClip для каждого видеофайла
    clips = [VideoFileClip(file_path) for file_path in file_paths]

    # Объединение видео в один клип
    final_clip = concatenate_videoclips(clips, method="compose")

    # Сохранение объединенного видеофайла
    output_file_path = f"concatenate.{file_paths[0].split('.')[-1]}"
    final_clip.write_videofile(output_file_path, codec="libx264", audio_codec="aac")

    return output_file_path 

st.set_page_config(layout="wide", page_title="Детекция с YOLO v4")
st.title("Детекция с YOLO v4")

st.sidebar.title('Информация и выбор режима')

app_mode = st.sidebar.selectbox(
    'Параметры',
    ['О сервисе', 'Изображение', 'Видео']
)

# About Page
if app_mode == 'О сервисе':
    st.markdown('''Проект создан с использованием фреймворка Streamlit и библиотеки OpenCV, при этом в его основе лежит сверточная нейросеть YOLOv4. 
    Проект предоставляет демонстрацию возможностей данной нейросети.    
#### YOLOv4 с обученными весами:
- Проект включает в себя готовую модель YOLOv4 с предварительно обученными весами. Эта модель способна обнаруживать объекты из 80 различных классов, что обеспечивает широкий спектр применения для определения различных объектов на изображениях.

#### Переобученная модель YOLOv4-tiny:
- Кроме того, представлена переобученная модель YOLOv4-tiny, которая была адаптирована для специфической задачи - определения наличия надетой каски на человеке. В ходе этого процесса были взяты веса базовой модели YOLOv4-tiny, проведена разметка данных и выполнено обучение на домашнем компьютере с использованием технологии CUDA от NVIDIA. Обучение велось по двум классам: "helmet" (каска) и "head" (голова).

Этот проект не только демонстрирует работу нейросети, но и подчеркивает его возможности в адаптации к конкретным задачам и обучению на собственных данных.
    ''')
    with st.expander("Пример видео на выходе после обработки"):
        st.video('processed_test1.mp4')
        st.video('processed_test2.mp4')

# Image Page
elif app_mode == 'Изображение':
    genre = st.sidebar.radio(
    "Выбор модели",
    ["yolov4", "helmet"],
    index=0,
    )
    if genre == 'yolov4':
        names = 'coco.names'
        weights = 'yolov4-tiny.weights'
        config = 'yolov4-tiny-custom.cfg'
    elif genre == 'helmet':
        names = 'helmet.names'
        weights = 'helmet_final.weights'
        config = 'helmet.cfg'
            
    st.sidebar.markdown('---')

    st.markdown("#### Определение объектов на изображении")

    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Порог уверенности', min_value=0.0, max_value=1.0, value=0.5)
    with st.sidebar.expander("Назначение:"):
        st.markdown('''Определяет, насколько уверенной должна быть модель в том, что объект обнаружен, прежде чем результат будет рассматриваться как положительный. 
                 Если уверенность модели ниже этого порога, объект не будет считаться детектированным.
                 Обычно устанавливается в диапазоне 0.1-0.9 в зависимости от ваших требований к уверенности. 
                 Более низкие значения приведут к более широкому набору детекций, но с большим числом ложных срабатываний.''')
    tracking_confidence = st.sidebar.slider('Порог подавления', min_value=0.0, max_value=1.0, value=0.5)
    with st.sidebar.expander("Назначение:"):
        st.markdown('''Используется для подавления (фильтрации) дублирующих детекций. 
                 Когда модель обнаруживает несколько прямоугольных областей, перекрывающихся с высокой уверенностью, порог подавления помогает выбрать только одну из них. 
                 Это предотвращает появление множественных детекций для одного и того же объекта.
                 Обычно устанавливается в диапазоне 0.1-0.5. 
                 Более высокие значения делают алгоритм более консервативным, выбирая более уверенные детекции, но при этом может пропустить менее уверенные детекции.''')
    st.sidebar.markdown('---')
    
    img_file_buffer = st.sidebar.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    
    
    st.sidebar.text('Исходное изображение')
    if img_file_buffer:
        st.sidebar.image(img_file_buffer)
    if img_file_buffer:
        start = st.button("Запуск обработки", type="primary")
        stframe = st.empty()
        if start:
            model, COLORS, class_name = yolov4(names, weights, config, detection_confidence, tracking_confidence)
            cap, temp_file_path = temp_file_image(img_file_buffer)
            frame = image_process(model, cap, detection_confidence, tracking_confidence, COLORS, class_name)
            stframe.image(frame,channels='BGR', use_column_width="auto")
# Video Page
elif app_mode == 'Видео':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    genre = st.sidebar.radio(
    "Выбор модели",
    ["yolov4", "helmet"],
    index=0,
    )
    
    if genre == 'yolov4':
        names = 'coco.names'
        weights = 'yolov4-tiny.weights'
        config = 'yolov4-tiny-custom.cfg'
    elif genre == 'helmet':
        names = 'helmet.names'
        weights = 'helmet_final.weights'
        config = 'helmet.cfg'
    
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Порог уверенности', min_value=0.0, max_value=1.0, value=0.5)
    with st.sidebar.expander("Назначение:"):
        st.markdown('''Определяет, насколько уверенной должна быть модель в том, что объект обнаружен, прежде чем результат будет рассматриваться как положительный. 
                 Если уверенность модели ниже этого порога, объект не будет считаться детектированным.
                 Обычно устанавливается в диапазоне 0.1-0.9 в зависимости от ваших требований к уверенности. 
                 Более низкие значения приведут к более широкому набору детекций, но с большим числом ложных срабатываний.''')
    tracking_confidence = st.sidebar.slider('Порог подавления', min_value=0.0, max_value=1.0, value=0.5)
    with st.sidebar.expander("Назначение:"):
        st.markdown('''Используется для подавления (фильтрации) дублирующих детекций. 
                 Когда модель обнаруживает несколько прямоугольных областей, перекрывающихся с высокой уверенностью, порог подавления помогает выбрать только одну из них. 
                 Это предотвращает появление множественных детекций для одного и того же объекта.
                 Обычно устанавливается в диапазоне 0.1-0.5. 
                 Более высокие значения делают алгоритм более консервативным, выбирая более уверенные детекции, но при этом может пропустить менее уверенные детекции.''')
    st.sidebar.markdown('---')

    ## Get Video
    
    video_file_buffer = st.sidebar.file_uploader("Загрузите видео", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    
    st.sidebar.text('Исходное видео')
    if video_file_buffer:
        st.sidebar.video(video_file_buffer)
        
    st.markdown("#### Определение объектов на видео")
    
    if video_file_buffer:
        start = st.button("Запуск обработки", type="primary")
        stop = st.button("Остановка обработки")
        stframe = st.empty()
        if start:
            model, COLORS, class_name = yolov4(names, weights, config, detection_confidence, tracking_confidence)
            temp_file_path = temp_file(video_file_buffer, video_file_buffer.name)
            created_files = split_video(temp_file_path, duration=10)
            for input_file in created_files:
                cap, temp_file_path = temp_file_for_process(input_file)
                video_frames = video_process(model, cap, detection_confidence, tracking_confidence, COLORS, class_name, input_file)
                concotinate_video(video_frames, temp_file_path, input_file)
            # final_clip = concatenate_videos_in_one(created_files)
            st.markdown('### Обработанное видео')
            with st.expander("Обработонное видео:"):
                st.markdown('''Так как streamlit cloud вводит ограничение на время выполнение операций,
                            сборка обработанных файлов невозможна в облаке''')
                for file in created_files:
                    st.video(file)
            
    
