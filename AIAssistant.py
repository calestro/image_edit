import torch
import numpy as np
import cv2
import os
from PIL import Image
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, EulerDiscreteScheduler, AutoencoderKL
from segment_anything import sam_model_registry, SamPredictor
import gc
import Routines
import image_to_image as m
from controlnet_aux import OpenposeDetector 
from transformers import pipeline, CLIPVisionModelWithProjection

# Configuração de Cache
CACHE_DIR = os.path.abspath("./models")
os.environ["HF_HOME"] = CACHE_DIR
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

class AIAssistant:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
   
        self.sam_checkpoint = m.SAM_CHECKPOINT 
        print(f"--- SAM Configurado: {self.sam_checkpoint} (Modo Gigante/Dinâmico) ---")
        self.load_models()

    def load_models(self):
        print(f"--- Cache definido para: {CACHE_DIR} ---")
        
        # 1. Carrega Image Encoder
        print("--- 1. Carregando Vision Encoder ---")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", 
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR
        ).to(self.device)

        # 2. Detectores
        print("--- 2. Carregando Detectores ---")
        self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir=CACHE_DIR)
        
        self.depth_estimator = pipeline(
            task="depth-estimation", 
            model="LiheYoung/depth-anything-small-hf", 
            device=0,
            model_kwargs={"cache_dir": CACHE_DIR}
        )

        print("--- 3. Carregando ControlNets ---")
        c_pose = ControlNetModel.from_pretrained("xinsir/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True, cache_dir=CACHE_DIR)
        c_depth = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True, cache_dir=CACHE_DIR)
        self.controlnets = [c_pose.to(self.device), c_depth.to(self.device)]
        
        # Limpeza
        gc.collect()
        torch.cuda.empty_cache()

        # 4. Pipeline Principal (SDXL)
        print("--- 4. Carregando SDXL (Isso consome a maior parte da RAM) ---")
       
        
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            m.MODEL_PATH,
            controlnet=self.controlnets,
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True, # Importante para não estourar a RAM
            cache_dir=CACHE_DIR,
            image_encoder=image_encoder,
        ).to(self.device)

       
        
       
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)
        
     
        self.pipe.enable_model_cpu_offload()        
        
  
        print("--- 5. Carregando IP-Adapter ---")
        try:
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter", 
                subfolder="sdxl_models", 
                weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                cache_dir=CACHE_DIR
            )
            if hasattr(self.pipe, "set_ip_adapter_scale"):
                self.pipe.set_ip_adapter_scale(0.6)
            print("--- IP-Adapter OK ---")
        except Exception as e:
            print(f"ERRO IP-Adapter: {e}")

       
        print("--- ✨ Sistema Pronto (Modo SAM Dinâmico) ---")

    def add_lora(self, filename, adapter_name):
        directory = f"loras/{filename}"
        if os.path.exists(directory):
            print(f"Carregando LoRA: {filename}")
            self.pipe.load_lora_weights(directory, adapter_name=adapter_name)
        else:
            print(f"LoRA não encontrado: {directory}")

    def get_mouse_clicks(self, image_pil):
        # Reduz a imagem para visualização se for muito grande
        w, h = image_pil.size
        scale = 1.0
        if max(w, h) > 1024:
            scale = 1024 / max(w, h)
            display_img = image_pil.resize((int(w*scale), int(h*scale)))
        else:
            display_img = image_pil

        cv_img = cv2.cvtColor(np.array(display_img), cv2.COLOR_RGB2BGR)
        points = []
        
        def callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                real_x = int(x / scale)
                real_y = int(y / scale)
                points.append((real_x, real_y))
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
        # --- CARREGAMENTO DO SAM GIGANTE SOB DEMANDA ---
        print(">>> Carregando SAM Huge (2.5GB) para memória...")
        
        # 1. Limpa VRAM antes de carregar o monstro
        gc.collect()
        torch.cuda.empty_cache()
        
   
        sam = sam_model_registry["vit_h"](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        predictor = SamPredictor(sam)

        print(">>> Gerando Máscara de Alta Precisão...")
        image_np = np.array(image_pil.convert("RGB"))
        predictor.set_image(image_np)
        
        input_points = np.array(points)
        input_labels = np.ones(len(points))
        
        masks, _, _ = predictor.predict(
            point_coords=input_points, 
            point_labels=input_labels, 
            multimask_output=False
        )
        
        mask_np = (masks[0] * 255).astype(np.uint8)
        
        # 3. DELETA O SAM IMEDIATAMENTE PARA NÃO TRAVAR A GERAÇÃO DEPOIS
        del predictor
        del sam
        gc.collect()
        torch.cuda.empty_cache()
        print(">>> SAM removido da memória. Espaço liberado para o SDXL.")
        
        # Pós-processamento da máscara
        kernel = np.ones((9, 9), np.uint8) 
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        mask_np = cv2.GaussianBlur(mask_np, (21, 21), 0)

        return Image.fromarray(mask_np)

    def prepare_pose_image(self, image_pil):
        print("Criando Pose...")
        pose_image = self.openpose_detector(image_pil)
        if pose_image.size != image_pil.size:
            pose_image = pose_image.resize(image_pil.size, Image.NEAREST)
        return pose_image

    def prepare_depth_image(self, image_pil):
        print("Criando Depth...")
        depth_map = self.depth_estimator(image_pil)["depth"]
        if depth_map.size != image_pil.size:
            depth_map = depth_map.resize(image_pil.size, Image.NEAREST)
        return depth_map
    
    def smooth_depth_map(self, depth_img, mask_img, kernel_size=61):
        depth_np = np.array(depth_img)
        mask_np = np.array(mask_img.resize(depth_img.size).convert("L"))
        blurred_depth = cv2.GaussianBlur(depth_np, (kernel_size, kernel_size), 0)
        mask_bool = mask_np > 128       
        depth_np[mask_bool] = blurred_depth[mask_bool]
        return Image.fromarray(depth_np)  

    def run(self):
        while True:
            print("\n" + "═"*40)
            print("  (1) Edição (SAM Huge + IP-Adapter)\n  (2) Sair")
            print("═"*40)
            modo = "1"
            if modo == "2": break
            
           
            
            Routines.exec(self)

if __name__ == "__main__":
    assistant = AIAssistant()
    assistant.run()