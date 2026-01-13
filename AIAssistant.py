import os
import torch
import numpy as np
import cv2
import gc
import Routines
import image_to_image as m
from PIL import Image

# --- CONFIGURAÇÃO DE CACHE LOCAL ---
CACHE_DIR = os.path.abspath("./models")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
# --------------------------------------------------------------------------

from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DPMSolverMultistepScheduler, AutoencoderKL
from segment_anything import sam_model_registry, SamPredictor
from controlnet_aux import OpenposeDetector 
from transformers import pipeline, CLIPVisionModelWithProjection # <--- IMPORTANTE: Import novo

class AIAssistant:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_models()

    def match_color_tone(self, target_img, source_img, mask_img):
        # ... (Mantido igual)
        try:  
            src = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR) 
            dst = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
            mask = np.array(mask_img.resize(target_img.size).convert("L"))
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(mask)
            if coords is None: return target_img
            x, y, w, h = cv2.boundingRect(coords)
            center = (x + w // 2, y + h // 2)
            output = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
            return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        except Exception as e:
            return Image.composite(target_img, source_img, mask_img)    
            
    def load_models(self):
        print(f"--- Cache definido para: {CACHE_DIR} ---")
        
        print("--- 1. Carregando Image Encoder (Necessário para IP-Adapter Plus) ---")
        # CORREÇÃO DO ERRO DE MATRIZ: Carregamos o encoder visual correto (ViT-H)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", 
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR
        ).to(self.device)

        print("--- 2. Carregando Detectores ---")
        self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir=CACHE_DIR)
        
        self.depth_estimator = pipeline(
            task="depth-estimation", 
            model="LiheYoung/depth-anything-small-hf", 
            device=0,
            model_kwargs={"cache_dir": CACHE_DIR}
        )

        print("--- 3. Carregando ControlNets ---")
        controlnet_pose = ControlNetModel.from_pretrained(
            "xinsir/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True, cache_dir=CACHE_DIR
        )
        controlnet_depth = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True, cache_dir=CACHE_DIR
        )
        self.controlnets = [controlnet_pose.to(self.device), controlnet_depth.to(self.device)]
        
        gc.collect()
        torch.cuda.empty_cache()

        print("--- 4. Carregando VAE ---")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, cache_dir=CACHE_DIR)

        print("--- 5. Carregando SDXL Inpaint com Encoder Injetado ---")
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            m.MODEL_PATH,
            controlnet=self.controlnets,
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            cache_dir=CACHE_DIR,
            image_encoder=image_encoder, # <--- AQUI ESTÁ A CORREÇÃO: Injetamos o encoder correto
        ).to(self.device)

        self.pipe.vae = vae.to(self.device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)
        self.pipe.enable_model_cpu_offload()        
        
        print("--- 6. Carregando SAM ---")
        sam = sam_model_registry["vit_h"](checkpoint=m.SAM_CHECKPOINT)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

        print("--- 7. Carregando IP-Adapter (Plus Face/Body) ---")
        try:
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter", 
                subfolder="sdxl_models", 
                weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                cache_dir=CACHE_DIR
            )
            # Define escala padrão
            if hasattr(self.pipe, "set_ip_adapter_scale"):
                self.pipe.set_ip_adapter_scale(0.6)
            print("--- IP-Adapter Carregado com Sucesso! ---")
        except Exception as e:
            print(f"ERRO ao carregar IP-Adapter: {e}")

        print("--- ✨ Tudo pronto! ---")

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
        masks, _, _ = self.sam_predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)
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
        return pose_image

    def prepare_depth_image(self, image_pil):
        print("Criando mapa de profundidade...")
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
            print("  (1) Edição Controlada (SAM + IP-Adapter + ControlNet)\n  (2) Sair")
            print("═"*40)
            modo = "1"
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