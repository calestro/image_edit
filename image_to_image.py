import os
import AIAssistant as AI


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"


MODEL_PATH = "./reality.safetensors"
SAM_CHECKPOINT = "./sam_vit_h_4b8939.pth"





MASTER_LORA = "./master.safetensors"
LORA_SKIN = "./lora_skin.safetensors"

if __name__ == "__main__":
    assistant = AI.AIAssistant()
    
    #assistant.add_lora(MASTER_LORA,"MASTER")
    #assistant.add_lora(LORA_SKIN,"SKIN")
    
    
    assistant.run()