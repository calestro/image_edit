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
                
                depth_img_raw = self.prepare_depth_image(init_img)
                depth_img = self.smooth_depth_map(depth_img_raw, mask_img, kernel_size=81)
                
                control_images_list = [pose_img, depth_img]
                total_width = pose_img.width + depth_img.width
                height = pose_img.height
                union_image = Image.new('RGB', (total_width, height))
                union_image.paste(pose_img, (0, 0))
                union_image.paste(depth_img, (pose_img.width, 0))

           
                union_image.save("union.png")
                print("Salvo debug: union.png")
                
              

                i = 1
                while(True):
                    
                    input_strength = input("Defina a força: ")

                    try:
                        strength = float(input_strength) 
                    except:
                        strength = 0.7

                    
                    try:
                        controlnet_GUIDE = float(input("FORÇA DO PROMPT: "))
                    except:
                        controlnet_GUIDE = 7


                    try:
                        profundidade = float(input("Profundidade: "))
                    except:
                        profundidade = .3    

                    control_guidance_end_list = [1.0, profundidade]
                    
                   
                    control_scales_list = [1.0, 0.8]
                
                    print(f"Editando... (Strength: {strength})")
                            
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
    