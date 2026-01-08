import torch
import numpy as np
import cv2

import os
from PIL import Image
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DPMSolverMultistepScheduler
from segment_anything import sam_model_registry, SamPredictor
from diffusers.utils import load_image
import gc
import Routines
import image_to_image as m
from controlnet_aux import OpenposeDetector 
from transformers import pipeline 

CONTROLNET_MODEL = "xinsir/controlnet-openpose-sdxl-1.0"

class AIAssistant:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_models()
        
    def match_color_tone(self, target_img, source_img, mask_img):
        """
        CORRIGIDO: Ajuste de Cor Preservando a Iluminação (Luminosity Preservation)
        Transfere a cor (A, B) da pele original, mas mantém a iluminação (L) criada pela IA.
        """
        print(">> Aplicando Correção de Cor (Color Only)...")
        
        # 1. Converter PIL -> Numpy -> LAB (Float32 para precisão)
        source_lab = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2LAB).astype("float32")
        target_lab = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2LAB).astype("float32")
        
        # 2. Configurar Máscaras
        mask = np.array(mask_img.resize(target_img.size).convert("L"))
        mask_binary = mask > 128  # Área gerada pela IA
        mask_inverted = mask <= 128 # Área original (Pele real)
        
        # Verificação de segurança
        if np.sum(mask_binary) == 0 or np.sum(mask_inverted) == 0:
            print("Mascara inválida ou vazia. Ignorando color match.")
            return target_img

        # 3. Calcular Médias
        # src_mean retorna array shape (3, 1) -> [L_mean, A_mean, B_mean]
        src_mean, _ = cv2.meanStdDev(source_lab, mask=mask_inverted.astype(np.uint8))
        tar_mean, _ = cv2.meanStdDev(target_lab, mask=mask.astype(np.uint8))

        # 4. Aplicar a Correção
        # Canal 0 = L (Luminosidade/Luz) -> NÃO MEXEMOS (ou mexemos muito pouco)
        # Canal 1 = A (Verde-Vermelho) -> Corrigimos para bater a cor
        # Canal 2 = B (Azul-Amarelo) -> Corrigimos para bater a cor
        
        # Separamos os canais para facilitar
        l, a, b = cv2.split(target_lab)

        # Calculamos a diferença apenas nas cores
        diff_a = src_mean[1][0] - tar_mean[1][0]
        diff_b = src_mean[2][0] - tar_mean[2][0]

        # Aplicamos a diferença APENAS onde a máscara diz (na pele nova)
        a[mask_binary] += diff_a
        b[mask_binary] += diff_b
        
        # Opcional: Ajuste leve no brilho (L) se a pele nova estiver MUITO escura/clara
        # Descomente a linha abaixo se quiser forçar um pouco o brilho original (ex: 50% de força)
        # l[mask_binary] += (src_mean[0][0] - tar_mean[0][0]) * 0.5

        # 5. Juntar os canais de volta
        result_lab = cv2.merge([l, a, b])

        # 6. Clip e Conversão Final
        result_lab = np.clip(result_lab, 0, 255).astype("uint8")
        result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        
        # 7. Blend Final (Suavizar as bordas do recorte)
        final_img = Image.fromarray(result_rgb)
        final_img = Image.composite(final_img, target_img, mask_img)
        
        return final_img
    
    
    
    
    
    def load_models(self):
        print("--- Carregando Detectores (OpenPose + Depth) ---")

        self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

        print("--- Carregando Detector de Profundidade (Depth Anything V2) ---")
        
        self.depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=0)

        print("--- Carregando Modelos ControlNet (OpenPose + Depth) ---")
      
        controlnet_pose = ControlNetModel.from_pretrained(
            "xinsir/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True,cache_dir="./models"
        )
        controlnet_depth = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True,cache_dir="./models"
        )

        self.controlnets = [controlnet_pose.to(self.device), controlnet_depth.to(self.device)]
        
        gc.collect()
        torch.cuda.empty_cache()

        print("--- Carregando SDXL Inpaint com Multi-ControlNet ---")
        # Passamos a LISTA de controlnets para o pipeline
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            m.MODEL_PATH,
            controlnet=self.controlnets, # Passando a lista aqui
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
         
        ).to(self.device)

        # ... o resto do scheduler, cpu offload e SAM continua igualzinho ...
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)
        self.pipe.enable_model_cpu_offload()        
        print("--- Carregando SAM ---")
        sam = sam_model_registry["vit_h"](checkpoint=m.SAM_CHECKPOINT)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        print("--- ✨ Tudo pronto (Modo Multi-ControlNet)! ---")

    def add_lora(self, filename, adapter_name):
        directory = f"loras/{filename}"
        if os.path.exists(directory):
            print(f"Carregando LoRA: {filename}")
            self.pipe.load_lora_weights(directory, adapter_name=adapter_name)
        else:
            print(f"LoRA não encontrado: {directory}")

    def get_mouse_clicks(self, image_pil):
        cv_img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        points = []
        
        def callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(cv_img, (x, y), 8, (0, 255, 0), -1)
                cv2.imshow("Editor IA", cv_img)
        
        cv2.namedWindow("Editor IA")
        cv2.setMouseCallback("Editor IA", callback)
        cv2.imshow("Editor IA", cv_img)
        print("Pressione ESC para terminar a seleção.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return points

    def generate_mask(self, image_pil, points):
        image_np = np.array(image_pil.convert("RGB"))
        self.sam_predictor.set_image(image_np)
        
        input_points = np.array(points)
        input_labels = np.ones(len(points))
        
        masks, _, _ = self.sam_predictor.predict(
            point_coords=input_points, 
            point_labels=input_labels, 
            multimask_output=False
        )
        
        mask_np = (masks[0] * 255).astype(np.uint8)
        
        kernel = np.ones((9, 9), np.uint8) 
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        mask_np = cv2.GaussianBlur(mask_np, (21, 21), 0)

        return Image.fromarray(mask_np)

    def prepare_pose_image(self, image_pil):
        print("Criando esqueleto da pose...")
        
  
        pose_image = self.openpose_detector(image_pil)
        
    
        if pose_image.size != image_pil.size:
            pose_image = pose_image.resize(image_pil.size, Image.NEAREST)
        
        pose_image.save("debug_pose.png")        
        return pose_image
    def prepare_depth_image(self, image_pil):
        print("Criando mapa de profundidade (Depth Anything)...")
        
        # O pipeline do transformers retorna um dicionário
        # Nós pegamos a chave "depth" que é a imagem processada
        depth_map = self.depth_estimator(image_pil)["depth"]
        
        # Garantir o tamanho correto (Bug do tensor)
        if depth_map.size != image_pil.size:
            depth_map = depth_map.resize(image_pil.size, Image.NEAREST)

        depth_map.save("debug_depth.png")        
        return depth_map

    def run(self):
        while True:
            print("\n" + "═"*40)
            print("  (1) Edição Controlada (SAM + OpenPose)\n  (2) Sair")
            print("═"*40)
            modo = "1"
            
            # modo = input("Escolha: ")
            
            if modo == "2": break

            nome_img = input("Nome do arquivo em ./assets/ (ex: g.jpg): ")
            img_path = f"./assets/{nome_img}"
            
            if not os.path.exists(img_path):
                print(f"Arquivo não encontrado: {img_path}")
                continue

            prompt = input("Prompt: ")
            user_neg = input("Negativo extra: ")
            neg = f"deformed, bad anatomy, missing limbs, blur, {user_neg}"
            
            Routines.exec(self, prompt, neg, img_path)

if __name__ == "__main__":
    assistant = AIAssistant()
    assistant.run()