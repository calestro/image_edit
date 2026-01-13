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
from diffusers import AutoencoderKL 

CONTROLNET_MODEL = "xinsir/controlnet-openpose-sdxl-1.0"


class AIAssistant:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_models()


    def match_color_tone(self, target_img, source_img, mask_img):
   
        print(">> Aplicando Seamless Clone (Mistura de Iluminação)...")

        try:  
            src = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR) 
            dst = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

            mask = np.array(mask_img.resize(target_img.size).convert("L"))
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            coords = cv2.findNonZero(mask)
            if coords is None:
                print("Máscara vazia, retornando imagem gerada.")
                return target_img
                
            x, y, w, h = cv2.boundingRect(coords)
            center = (x + w // 2, y + h // 2)

            output = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)

            result = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            return result

        except Exception as e:
            print(f"Erro no Blending (provavelmente borda da imagem): {e}")
            return Image.composite(target_img, source_img, mask_img)    
            
    
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

        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

        print("--- Carregando SDXL Inpaint com Multi-ControlNet ---")
     
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            m.MODEL_PATH,
            controlnet=self.controlnets, # Passando a lista aqui
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
           
         
        ).to(self.device)

        self.pipe.vae = vae.to(self.device)
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
    
    def smooth_depth_map(self, depth_img, mask_img, kernel_size=61):

        depth_np = np.array(depth_img)
        mask_np = np.array(mask_img.resize(depth_img.size).convert("L"))
        
        blurred_depth = cv2.GaussianBlur(depth_np, (kernel_size, kernel_size), 0)
        
        mask_bool = mask_np > 128       

        if len(depth_np.shape) == 3:
            depth_np[mask_bool] = blurred_depth[mask_bool]
        else:
            depth_np[mask_bool] = blurred_depth[mask_bool]
            
        return Image.fromarray(depth_np)        

if __name__ == "__main__":
    assistant = AIAssistant()
    assistant.run()