from PIL import Image
import datetime
import os
import numpy as np # Necessário para cálculos de máscara

def exec(self):
    
    try:
        while(True):  
            print("\n" + "═"*50)
            nome_img = input("Nome do arquivo em ./assets/ (ex: g.jpg): ")
            img_path = f"./assets/{nome_img}"
            if not os.path.exists(img_path):
                print(f"❌ Arquivo não encontrado: {img_path}")
                continue
           
            prompt = input("Prompt: ")
            user_neg = input("Negativo extra: ")
            neg = f"deformed, bad anatomy, missing limbs, blur, {user_neg}"
            
            # Carrega imagem original
            init_img = Image.open(img_path).convert("RGB").resize((1024, 1024))                        

            # --- 1. Configuração do MODO do IP-Adapter ---
            print("\n--- 🤖 Configuração do IP-Adapter (Referência) ---")
            print("   [ENTER] = Modo RECORTE (Foca no detalhe. Melhor para rostos/manter estilo).")
            print("   ['inv'] = Modo INVERSO (Oculta a seleção. Melhor para TROCAR roupas/objetos).")
            print("   [Caminho] = Usa uma imagem externa como referência.")
            ref_input = input(">> Escolha: ")

            ip_image = None
            mode_ip = "crop" # padrão

            # Se for caminho de arquivo externo, já carrega agora
            if ref_input.strip() != "" and ref_input != "inv":
                if os.path.exists(ref_input):
                    ip_image = Image.open(ref_input).convert("RGB").resize((1024, 1024))
                    mode_ip = "external"
                    print(f"✅ Referência Externa carregada: {ref_input}")
                else:
                    print("⚠️ Caminho inválido. IP-Adapter será desligado ou usará fallback.")
            elif ref_input == "inv":
                mode_ip = "inverse"
            else:
                mode_ip = "crop"

            # --- 2. Seleção e Máscara ---
            print("\n[INFO] Clique na área que deseja mudar.")
            points = self.get_mouse_clicks(init_img)
            if not points: break # Sai se não houver pontos

            print("⏳ Gerando Máscara...")
            mask_img = self.generate_mask(init_img, points)
            mask_img.save("mask.png")

            # --- 3. Processamento Avançado do IP-Adapter (Pós-Máscara) ---
            
            # Caso A: Modo INVERSO (Sua ideia para trocar calça por shorts)
            if mode_ip == "inverse":
                print(">> 🔄 Aplicando Modo INVERSO (Contexto externo)...")
                # Cria fundo preto
                black_bg = Image.new("RGB", init_img.size, (0, 0, 0))
                # Onde a máscara é branca (seleção), fica preto. O resto mantém a imagem original.
                ip_image = Image.composite(black_bg, init_img, mask_img)
                # Redimensiona para o CLIP ver melhor
                ip_image = ip_image.resize((1024, 1024))
                ip_image.save("debug_ip_inv.png")
                print("   (Debug salvo em debug_ip_inv.png)")

            # Caso B: Modo RECORTE (Melhor para inpainting de rostos/correções)
            elif mode_ip == "crop":
                print(">> 🔍 Aplicando Modo RECORTE (Foco no detalhe)...")
                mask_arr = np.array(mask_img.convert("L"))
                where_mask = np.where(mask_arr > 0)

                if where_mask[0].size > 0:
                    # Calcula bounding box
                    y1, x1 = np.min(where_mask, axis=1)
                    y2, x2 = np.max(where_mask, axis=1)
                    
                    # Padding de segurança
                    pad = 30
                    y1 = max(0, y1 - pad); x1 = max(0, x1 - pad)
                    y2 = min(mask_arr.shape[0], y2 + pad); x2 = min(mask_arr.shape[1], x2 + pad)
                    
                    # Corta a imagem original na área da máscara
                    ip_image = init_img.crop((x1, y1, x2, y2))
                    ip_image = ip_image.resize((512, 512)) # Tamanho ideal para o encoder
                    ip_image.save("debug_ip_crop.png")
                    print("   (Debug salvo em debug_ip_crop.png)")
                else:
                    print("⚠️ Máscara vazia detectada. Usando imagem total.")
                    ip_image = init_img

            # Loop para refazer máscara se necessário
            # (Removi o loop complexo de máscara aqui para simplificar, 
            #  mas se quiser refazer a máscara teria que voltar ao inicio ou encapsular lógica)
            
            # --- 4. ControlNet Prep ---
            print("⚙️ Criando controles (Pose + Depth)...")
            pose_img = self.prepare_pose_image(init_img)
            
            depth_img_raw = self.prepare_depth_image(init_img)
            # Smooth depth ajuda a mesclar melhor as bordas
            depth_img = self.smooth_depth_map(depth_img_raw, mask_img, kernel_size=81)
            
            control_images_list = [pose_img, depth_img]
            
            # Debug Visual
            total_width = pose_img.width + depth_img.width
            height = pose_img.height
            union_image = Image.new('RGB', (total_width, height))
            union_image.paste(pose_img, (0, 0))
            union_image.paste(depth_img, (pose_img.width, 0))
            union_image.save("union.png")
            print("   (Debug controles salvo em union.png)")

            # --- 5. Loop de Geração (Ajuste de Parâmetros) ---
            i = 1
            while(True):
                print(f"\n--- Geração #{i} ---")
                
                # Inputs com valores padrão robustos
                s_input = input("Denoising Strength (0.6 - 1.0) [0.7]: ")
                strength = float(s_input) if s_input else 0.7

                g_input = input("CFG Scale (Prompt) [7.0]: ")
                controlnet_GUIDE = float(g_input) if g_input else 7.0

                ip_input = input("Força IP-Adapter (0.0 - 1.0) [0.6]: ")
                ip_scale = float(ip_input) if ip_input else 0.6
                
                depth_start_input = input("Força depth_start-Adapter (0.0 - 1.0) [0.6]: ")
                depth_start_scale = float(depth_start_input) if depth_start_input else 0.6

                depth_end_input = input("Força depth_end-Adapter (0.0 - 1.0) [0.6]: ")
                depth_end_scale = float(depth_end_input) if depth_end_input else 0.6
                

                # Configura IP-Adapter Scale dinamicamente
                if hasattr(self.pipe, "set_ip_adapter_scale"):
                    self.pipe.set_ip_adapter_scale(ip_scale)

                print(f"🚀 Editando... (Mode: {mode_ip} | Str: {strength} | IP: {ip_scale})")
                
                # Execução
                output = self.pipe(
                 prompt=prompt,
                 negative_prompt=neg,
                 image=init_img,
                 mask_image=mask_img,
                 control_image=control_images_list, 
                 controlnet_conditioning_scale=[1.0, depth_start_scale], # Pose Forte, Depth Médio
                 control_guidance_end=[1.0, depth_end_scale],          # Pose até o fim, Depth solta antes
                 num_inference_steps=30,
                 guidance_scale=controlnet_GUIDE,
                 strength=strength,
                 ip_adapter_image=ip_image 
                ).images[0]
        
                filename = f"output_{datetime.datetime.now().strftime('%H%M%S')}_{i}.png"
                output_path = f"./output/{filename}"
                
                # Cria pasta output se não existir
                if not os.path.exists("./output"): os.makedirs("./output")
                
                output.save(output_path)
                print(f"✨ Salvo: {output_path}")

                refresh = input("\n[1] Refazer parâmetros | [Enter] Nova Imagem: ")
                if(refresh != "1"):
                    break
                i = i + 1
                
    except Exception as e:
        print(f"\n❌ Erro Fatal: {e}")
        import traceback
        traceback.print_exc()