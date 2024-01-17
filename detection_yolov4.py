import cv2 as cv
import streamlit as st
import tempfile
import time


def yolov4(names, weights, config, data, Conf_threshold, NMS_threshold):
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
    
    # Создание контейнера для видео в Streamlit
    video_container = st.empty()
    video_frames = []
    
    # Сохранение видеофайла во временный файл
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(data.read())
    temp_file_path = temp_file.name
    
    # Захват видео
    cap = cv.VideoCapture(temp_file_path)
    starting_time = time.time()
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        frame_counter += 1
        if ret == False:
            break
        # Обнаружение объектов с использованием YOLOv4
        if frame_counter % 5 == 0:
            classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
            for (classid, score, box) in zip(classes, scores, boxes):
                print(classid)
                color = COLORS[int(classid) % len(COLORS)]
                label = "%s : %f" % (class_name[classid], score)
                cv.rectangle(frame, box, color, 1)
                cv.putText(frame, label, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
            if app_mode == 'Видео':
                endingTime = time.time() - starting_time
                fps = frame_counter/endingTime
                cv.putText(frame, f'FPS: {fps}', (20, 50),
                        cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            
            # Преобразование кадра в формат RGB для отображения в Streamlit
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            video_frames.append(frame.copy())
            
            # Отображение кадра в контейнере Streamlit
            video_container.image(frame, channels="RGB")
        
        stop = st.sidebar.radio(
        "Остановить обработку",
        ["Стоп"],
        index=None,
        )
        if stop:
            break
    cap.release()
    # cv.destroyAllWindows()
    
    if frame is not None and not frame.empty():
        frame = cv.resize(frame,(0,0), fx=0.8, fy=0.8)
        frame = image_resize(image=frame, width=640)
        stframe.image(frame,channels='BGR', use_column_width=True)
    

st.set_page_config(layout="wide", page_title="Детекция с YOLO v4")
st.title("Детекция с YOLO v4")

st.sidebar.title('Информация и выбор режима')

app_mode = st.sidebar.selectbox(
    'Параметры',
    ['О сервисе', 'Изображение', 'Видео']
)


@st.cache()
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # Получаем высоту (h) и ширину (w) изображения
    (h, w) = image.shape[:2]
    
    # Если не указаны новые размеры (width и height), возвращаем исходное изображение
    if width is None and height is None:
        return image
    
    # Если не указана новая ширина (width), вычисляем новые размеры на основе высоты (с сохранением соотношения сторон)
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    # Иначе, если не указана новая высота (height), вычисляем новые размеры на основе ширины (с сохранением соотношения сторон)
    else:
        r = width / float(w)
        dim = width, int(h * r)

    # Изменяем размер изображения с использованием указанных размеров и метода интерполяции
    # INTER_AREA - метод интерполяции для уменьшения изображения
    resized = cv.resize(image, dim, interpolation=inter)

    # Возвращаем измененное изображение
    return resized




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

    detection_confidence = st.sidebar.slider('Порог уверенности', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Порог подавления', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')
    
    img_file_buffer = st.sidebar.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    
    
    st.sidebar.text('Исходное изображение')
    if img_file_buffer:
        st.sidebar.image(img_file_buffer)
    if img_file_buffer:
        start = st.button("Запуск обработки", type="primary")
        if start:
            yolov4(names, weights, config, img_file_buffer, detection_confidence, tracking_confidence)
    

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
    tracking_confidence = st.sidebar.slider('Порог подавления', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    ## Get Video
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Загрузите видео", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    
    st.sidebar.text('Исходное видео')
    if video_file_buffer:
        st.sidebar.video(video_file_buffer)
        
    st.markdown("#### Определение объектов на изображении")
    
    if video_file_buffer:
        start = st.button("Запуск обработки", type="primary")
        if start:
            yolov4(names, weights, config, video_file_buffer, detection_confidence, tracking_confidence)
    
