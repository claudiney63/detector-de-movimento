import cv2
import numpy as np
import os
from tqdm import tqdm
import time

# Configuração inicial
VIDEO_PATH = 'video.mp4'  # Caminho para o vídeo de entrada
OUTPUT_DIR = 'output/'   # Diretório para salvar os resultados

# Criar diretório de saída, se não existir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Função para calcular fluxo óptico usando Farneback
def calculate_optical_flow_farneback(gray1, gray2):
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow

# Função para calcular fluxo óptico usando Lucas-Kanade
def calculate_optical_flow_lucas_kanade(frame1, frame2):
    # Parâmetros para o método Lucas-Kanade
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Detectar pontos de interesse usando Shi-Tomasi
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    points1 = cv2.goodFeaturesToTrack(gray1, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)
    
    # Calcular o fluxo óptico
    points2, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, points1, None, **lk_params)
    
    # Criar uma máscara para visualizar o fluxo
    mask = np.zeros_like(frame1)
    for i, (p1, p2) in enumerate(zip(points1, points2)):
        if status[i] == 1:
            x1, y1 = p1.ravel()
            x2, y2 = p2.ravel()
            mask = cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            frame2 = cv2.circle(frame2, (int(x2), int(y2)), 5, (0, 0, 255), -1)
    
    return cv2.add(frame2, mask)

# Função para calcular fluxo óptico usando Horn-Schunck
def calculate_optical_flow_horn_schunck(gray1, gray2, alpha=10, iterations=100):
    # Inicializar os campos de fluxo
    u = np.zeros_like(gray1, dtype=np.float32)
    v = np.zeros_like(gray2, dtype=np.float32)

    # Calcular as derivadas usando filtros de Sobel
    fx = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(gray2, cv2.CV_32F, 1, 0, ksize=3)
    fy = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3) + cv2.Sobel(gray2, cv2.CV_32F, 0, 1, ksize=3)
    ft = gray2.astype(np.float32) - gray1.astype(np.float32)

    # Iterar para calcular o fluxo óptico
    for _ in range(iterations):
        u_avg = cv2.boxFilter(u, -1, (3, 3))
        v_avg = cv2.boxFilter(v, -1, (3, 3))
        numerator = (fx * u_avg + fy * v_avg + ft)
        denominator = (alpha ** 2 + fx ** 2 + fy ** 2)
        u = u_avg - fx * numerator / denominator
        v = v_avg - fy * numerator / denominator

    return u, v

# Função principal para processar o vídeo
def process_video(video_path, method):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    
    if not ret:
        print("Erro ao abrir o vídeo.")
        return
    
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_method_dir = os.path.join(OUTPUT_DIR, method)
    
    if not os.path.exists(output_method_dir):
        os.makedirs(output_method_dir)

    # Configuração da barra de progresso
    start_time = time.time()
    with tqdm(total=total_frames, desc=f"Processando ({method})") as pbar:
        while True:
            ret, frame2 = cap.read()
            if not ret:
                break

            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            if method == 'farneback':
                flow = calculate_optical_flow_farneback(gray1, gray2)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros_like(frame1)
                hsv[..., 1] = 255
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                output_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            elif method == 'lucas-kanade':
                output_frame = calculate_optical_flow_lucas_kanade(frame1, frame2)
            elif method == 'horn-schunck':
                u, v = calculate_optical_flow_horn_schunck(gray1, gray2)
                mag, ang = cv2.cartToPolar(u, v)
                hsv = np.zeros_like(frame1)
                hsv[..., 1] = 255
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                output_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            else:
                print("Método inválido!")
                break

            # Salvar frame
            cv2.imwrite(f"{output_method_dir}/frame_{frame_count:04d}.png", output_frame)
            pbar.update(1)
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / (frame_count + 1)) * total_frames
            pbar.set_postfix({"ETA (s)": f"{estimated_total_time - elapsed_time:.2f}"})

            gray1 = gray2
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processamento concluído! Frames salvos em {output_method_dir}.")

# Escolha do método
print("Escolha o método de fluxo óptico:")
print("1 - Farneback")
print("2 - Lucas-Kanade")
print("3 - Horn-Schunck")
choice = input("Digite o número do método desejado: ")

if choice == '1':
    process_video(VIDEO_PATH, 'farneback')
elif choice == '2':
    process_video(VIDEO_PATH, 'lucas-kanade')
elif choice == '3':
    process_video(VIDEO_PATH, 'horn-schunck')
else:
    print("Escolha inválida!")
