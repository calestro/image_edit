import torch
import numpy as np
import cv2
import datetime
import os
from PIL import Image
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DPMSolverMultistepScheduler
from segment_anything import sam_model_registry, SamPredictor
from diffusers.utils import load_image


MODEL_PATH = "reality.safetensors"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MASTER_LORA = "master.safetensors"
CONTROLNET_MODEL = "diffusers/controlnet-canny-sdxl-1.0" 

class AIAssistant:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_models()
        
    def load_models(self):
        print("---Carregando ControlNet ---")
        self.controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(self.device)

        print("---Carregando SDXL Inpaint com ControlNet ---")
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            MODEL_PATH,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(self.device)

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, 
            use_karras_sigmas=True
        )
        
        self.pipe.enable_model_cpu_offload()

        if os.path.exists(MASTER_LORA):
            self.pipe.load_lora_weights(MASTER_LORA)
            self.pipe.fuse_lora(lora_scale=0.7)

        print("--- Carregando SAM (Segment Anything) ---")
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        print("--- ✨ Tudo pronto! ---")

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
        kernel = np.ones((15, 15), np.uint8) # Dilatação leve
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        return Image.fromarray(mask_np)

    def prepare_canny_image(self, image):
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

    def run(self):
        while True:
            print("\n" + "═"*40)
            print("  (1) Edição Controlada (SAM + ControlNet)\n  (2) Sair")
            print("═"*40)
            #####VOU ADICIONAR MAIS MODOS
            modo = 1
            if modo == "2": break

            img_path = input("Caminho da imagem: ")
            if not os.path.exists(img_path):
                print("Arquivo não encontrado.")
                continue

            prompt = input("Prompt (ex: blue shorts, denim texture): ")
            user_neg = input("Negativo extra: ")
            neg = f"deformed, bad anatomy, missing limbs, blur, {user_neg}"

            try:
                init_img = Image.open(img_path).convert("RGB").resize((1024, 1024))                        
                print("\n[INFO] Clique na área que deseja mudar.")
                points = self.get_mouse_clicks(init_img)
                if not points: continue
               
                print("Gerando Máscara...")
                mask_img = self.generate_mask(init_img, points)

               
                print("Criando esqueleto")
                control_image = self.prepare_canny_image(init_img)
               
                control_image.save("debug_canny.png") 

                input_strength = input("Defina a força do Control net: ")

                try:
                    strength = float(input_strength) 
                except:
                    strength = 0.8


                controlnet_scale = 0.5
                
                print(f"Editando... (Strength: {strength}, Control: {controlnet_scale})")
                
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    image=init_img,
                    mask_image=mask_img,
                    control_image=control_image, 
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    strength=strength, 
                    controlnet_conditioning_scale=controlnet_scale
                ).images[0]

                filename = f"edit_{datetime.datetime.now().strftime('%H%M%S')}.png"
                output.save(filename)
                print(f"Salvo: {filename}")

            except Exception as e:
                print(f"Erro: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    assistant = AIAssistant()
    assistant.run()