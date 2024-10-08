import cv2
import numpy as np
import os

class VideoMotionDetector:
    def __init__(self, video_path, width=640, height=780, save_interval=50):
        self.video_path = video_path
        self.width = width
        self.height = height
        self.save_interval = save_interval  # Intervalo entre as imagens salvas
        self.cap = cv2.VideoCapture(video_path)
        self.frame1 = None
        self.frame2 = None
        self.result_folder = "result"
        self.image_count = 0
        self.frame_count = 0

        # Verificar se o vídeo foi carregado corretamente
        if not self.cap.isOpened():
            raise Exception("Erro ao abrir o vídeo")

        # Criar pasta para resultados, se não existir
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)

        # Ler e redimensionar os primeiros quadros
        self.init_frames()

    def init_frames(self):
        """Lê e redimensiona os quadros iniciais do vídeo."""
        ret, self.frame1 = self.cap.read()
        ret, self.frame2 = self.cap.read()
        if ret:
            self.frame1 = cv2.resize(self.frame1, (self.width, self.height))
            self.frame2 = cv2.resize(self.frame2, (self.width, self.height))

    def detect_motion(self):
        """Realiza a detecção de movimento e gera uma imagem com efeito colorido."""
        diff = cv2.absdiff(self.frame1, self.frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Aplicar limiarização para binarizar a imagem
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        # Criar uma máscara colorida para as regiões com movimento detectado
        colored_mask = np.zeros_like(self.frame1)
        colored_mask[thresh == 255] = [0, 0, 255]

        # Combinar a máscara com o quadro original
        highlighted_frame = cv2.addWeighted(self.frame1, 0.7, colored_mask, 0.3, 0)
        return highlighted_frame

    def save_image(self, image):
        """Salva a imagem final na pasta result."""
        image_path = f"{self.result_folder}/motion_{self.image_count}.png"
        cv2.imwrite(image_path, image)
        print(f"Imagem salva: {image_path}")
        self.image_count += 1

    def start_detection(self):
        """Inicia o loop de detecção de movimento e exibição do vídeo."""
        # Configurar a posição da janela no meio da tela
        screen_width, screen_height = 1920, 1080
        x_position = (screen_width - self.width) // 2
        y_position = (screen_height - self.height) // 2

        # Criar uma janela e posicioná-la no centro
        cv2.namedWindow('Detecta Movimento com Efeito Colorido', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Detecta Movimento com Efeito Colorido', x_position, y_position)

        # Loop principal para reprodução e detecção
        while cv2.getWindowProperty('Detecta Movimento com Efeito Colorido', cv2.WND_PROP_VISIBLE) >= 1:
            # Verificar se o vídeo terminou
            if not self.cap.isOpened() or self.frame1 is None or self.frame2 is None:
                break

            # Detectar movimento e gerar imagem com destaque
            highlighted_frame = self.detect_motion()

            # Exibir o quadro com o movimento destacado
            cv2.imshow('Detecta Movimento com Efeito Colorido', highlighted_frame)

            # Salvar imagens em intervalos específicos de frames
            if self.frame_count % self.save_interval == 0:
                self.save_image(highlighted_frame)

            # Incrementar o contador de frames
            self.frame_count += 1

            # Preparar o próximo quadro
            self.frame1 = self.frame2
            ret, self.frame2 = self.cap.read()

            if ret:
                self.frame2 = cv2.resize(self.frame2, (self.width, self.height))
            else:
                # Reiniciar o vídeo se ele terminou
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.init_frames()

            # Checar se a janela foi fechada manualmente
            if cv2.waitKey(1) == 27:  # ESC para sair, se necessário
                break

        # Liberar os recursos e fechar a janela
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Processamento concluído! Imagens salvas na pasta '{self.result_folder}'.")

# Executar o detector com um exemplo de vídeo
if __name__ == "__main__":
    video_path = 'video.mp4'
    detector = VideoMotionDetector(video_path, save_interval=50)  # Salva a cada 50 frames
    detector.start_detection()
