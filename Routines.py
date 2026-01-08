from PIL import Image
import datetime

def exec(self,prompt,neg,image_path):
            try:
                init_img = Image.open(image_path).convert("RGB").resize((1024, 1024))                        
                while(True):    
                    print("\n[INFO] Clique na área que deseja mudar.")
                    points = self.get_mouse_clicks(init_img)
                    if not points: exit
               
                    print("Gerando Máscara...")
                
                    mask_img = self.generate_mask(init_img, points)
                    mask_img.save("mask.png")

                    input_mask = input("(1) Refazer Mascara: ")
                    if(input_mask != "1"):
                         break

               
                print("Criando esqueleto")
             
                pose_img = self.prepare_pose_image(init_img)
                depth_img = self.prepare_depth_image(init_img)
    

                control_images_list = [pose_img, depth_img]
                total_width = pose_img.width + depth_img.width
                height = pose_img.height
                union_image = Image.new('RGB', (total_width, height))
                union_image.paste(pose_img, (0, 0))
                union_image.paste(depth_img, (pose_img.width, 0))

           
                union_image.save("union.png")
                print("Salvo debug: union.png")
                control_scales_list = [0.7, 0.5]

                i = 1
                while(True):
                    
                    input_strength = input("Defina a força: ")

                    try:
                        strength = float(input_strength) 
                    except:
                        strength = 0.7

                    try:
                        controlnet_scale = float(input("controlnet GUIA SCALE: "))
                    except:
                     controlnet_scale = .65    
                    try:
                        controlnet_GUIDE = float(input("CONTROLE DO PROMPT: "))
                    except:
                        controlnet_GUIDE = 7


               
                
                    print(f"Editando... (Strength: {strength}, Control: {controlnet_scale})")
                            
                    output = self.pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    image=init_img,
                    mask_image=mask_img,
                    control_image=control_images_list, 
                    controlnet_conditioning_scale=control_scales_list,
                    num_inference_steps=50,
                    guidance_scale=controlnet_GUIDE,
                    strength=strength
                ).images[0]
                    
            #        try:
            #            output = self.match_color_tone(output, init_img, mask_img)
            #            print(">> Correção de cor aplicada com sucesso.")
            #        except Exception as e:
            #            print(f"AVISO: Falha na correção de cor: {e}")                             
            
                    filename = f"output_{datetime.datetime.now().strftime('%H%M%S')} ({str(i)}).png"
                    output.save(f"./output/{filename}")
                    print(f"Salvo: {filename}")

                    refresh = input("(1)Quer Refazer: ")

                    if(refresh != "1"):
                        break
                    i = i + 1
            except Exception as e:
                print(f"Erro: {e}")
                import traceback
                traceback.print_exc()
    