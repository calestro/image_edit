from PIL import Image
import datetime
import os

def exec(self):
    
    try:
        while(True):  
            nome_img = input("Nome do arquivo em ./assets/ (ex: g.jpg): ")
            img_path = f"./assets/{nome_img}"
            if not os.path.exists(img_path):
                print(f"Arquivo não encontrado: {img_path}")
           
                
            prompt = input("Prompt: ")
            user_neg = input("Negativo extra: ")
            neg = f"deformed, bad anatomy, missing limbs, blur, {user_neg}"
            init_img = Image.open(img_path).convert("RGB").resize((1024, 1024))                        

            # --- NOVO: Configuração da Imagem de Referência (IP-Adapter) ---
            ip_image = None
            print("\n--- Configuração do IP-Adapter (Cópia de Estilo/Pele) ---")
            ref_path = input("Caminho da imagem de REFERÊNCIA (Deixe vazio para usar a própria imagem original): ")

            if ref_path.strip() == "":
                print("Usando a própria imagem original como referência (preserva identidade/pele).")
                ip_image = init_img
            elif os.path.exists(ref_path):
                ip_image = Image.open(ref_path).convert("RGB").resize((1024, 1024))
                print(f"Referência carregada: {ref_path}")
            else:
                print("Caminho inválido. IP-Adapter será DESATIVADO nesta geração.")
                ip_image = None
            # ---------------------------------------------------------------


            print("\n[INFO] Clique na área que deseja mudar.")
            points = self.get_mouse_clicks(init_img)
            if not points: exit

            print("Gerando Máscara...")
            mask_img = self.generate_mask(init_img, points)
            mask_img.save("mask.png")

            input_mask = input("(1) Refazer Mascara | (Enter) Continuar: ")
            if(input_mask == "1"):
                continue
            
            print("Criando controles (Pose + Depth)...")
            pose_img = self.prepare_pose_image(init_img)
            
            depth_img_raw = self.prepare_depth_image(init_img)
            depth_img = self.smooth_depth_map(depth_img_raw, mask_img, kernel_size=81)
            
            control_images_list = [pose_img, depth_img]
            
            # Debug visual dos controles
            total_width = pose_img.width + depth_img.width
            height = pose_img.height
            union_image = Image.new('RGB', (total_width, height))
            union_image.paste(pose_img, (0, 0))
            union_image.paste(depth_img, (pose_img.width, 0))
            union_image.save("union.png")
            print("Salvo debug: union.png")

            i = 1
            while(True):
                input_strength = input("Denoising Strength (0.6 - 1.0) [Padrão 0.7]: ")
                try: strength = float(input_strength) 
                except: strength = 0.7

                input_guide = input("CFG Scale (Prompt) [Padrão 7.0]: ")
                try: controlnet_GUIDE = float(input_guide)
                except: controlnet_GUIDE = 7.0

                input_ip_scale = input("Força do IP-Adapter (0.0 - 1.0) [Padrão 0.6]: ")
                try: ip_scale = float(input_ip_scale)
                except: ip_scale = 0.6
                
                try: startGuidance = float( input("Valor da Guia de Profundidade Start [Padrão 0.6]: "))
                except: startGuidance = 0.6

                try: endGuidance = float(input("Valor da Guia de Profundidade END [Padrão 0.3]: "))
                except: endGuidance = 0.3

                control_guidance_end_list = [1.0, endGuidance] 
                control_scales_list = [1.0, startGuidance]
            
                print(f"Editando... (Strength: {strength} | IP-Adapter: {ip_scale})")
                
                # Configura a força do IP-Adapter dinamicamente
                if ip_image is not None:
                    self.pipe.set_ip_adapter_scale(ip_scale)
                        
                output = self.pipe(
                 prompt=prompt,
                 negative_prompt=neg,
                 image=init_img,
                 mask_image=mask_img,
                 control_image=control_images_list, 
                 controlnet_conditioning_scale=control_scales_list,
                 control_guidance_end=control_guidance_end_list, 
                 num_inference_steps=30,
                 guidance_scale=controlnet_GUIDE,
                 strength=strength,
                 ip_adapter_image=ip_image 
                ).images[0]
                
                # O match_color_tone antigo (seamlessClone) pode atrapalhar o IP-Adapter,
                # então deixei comentado. Se quiser usar, descomente.
                # try:
                #    output = self.match_color_tone(output, init_img, mask_img)
                # except Exception as e: print(e)
        
                filename = f"output_{datetime.datetime.now().strftime('%H%M%S')} ({str(i)}).png"
                output.save(f"./output/{filename}")
                print(f"Salvo: {filename}")

                refresh = input("(1) Refazer com novos parâmetros | (Enter) Nova seleção: ")
                if(refresh != "1"):
                    break
                i = i + 1
    except Exception as e:
        print(f"Erro Fatal: {e}")
        import traceback
        traceback.print_exc()