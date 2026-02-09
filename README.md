# 🎨 AI Image Editor (SDXL + ControlNet + SAM)

Um editor de imagens local poderoso impulsionado por Inteligência Artificial, combinando Stable Diffusion XL (SDXL), ControlNet, Segment Anything Model (SAM) e IP-Adapter.

Este projeto permite edições avançadas de imagens (Inpainting / Outpainting) com altíssima precisão e controle, possibilitando trocar roupas, objetos ou fundos mantendo iluminação, perspectiva e geometria da imagem original.

---

## ✨ Funcionalidades

### 🖱️ Segmentação Interativa (SAM)
Utiliza o Segment Anything Model (SAM) para criar máscaras extremamente precisas apenas clicando nas áreas desejadas da imagem.

### 🎨 SDXL Inpainting
Geração de imagens baseada no Stable Diffusion XL, otimizada para edições locais de alta qualidade.

### 🧭 ControlNet Integration
Mantém a pose, profundidade e estrutura da cena, garantindo que as edições respeitem a geometria original da imagem.

### 🧩 IP-Adapter (Image Prompt Adapter)

- Modo Recorte  
  Foca nos detalhes da área mascarada (ideal para rostos).

- Modo Inverso (inv)  
  Permite substituições completas (ex: trocar roupas), ignorando o conteúdo original da máscara.

- Referência Externa  
  Use uma imagem externa para transferir estilo, cores ou características visuais.

### 🎭 Suporte a LoRA
Carregamento simples de múltiplos LoRAs para estilização avançada.

### 🧠 Gerenciamento Inteligente de Memória
O modelo SAM (ViT-H) é carregado apenas quando necessário, liberando VRAM para a geração com SDXL.

---

## 🛠️ Pré-requisitos

- Python 3.10+
- GPU NVIDIA com suporte a CUDA
- Recomendado: 12GB+ de VRAM
- Drivers CUDA instalados corretamente

---

## 📦 Instalação

Clone o repositório:

git clone https://github.com/seu-usuario/seu-projeto.git  
cd seu-projeto

Crie um ambiente virtual (opcional, mas recomendado):

python -m venv venv  
Windows: .\venv\Scripts\activate  
Linux/Mac: source venv/bin/activate

Instale as dependências criando um arquivo requirements.txt com o conteúdo abaixo:

torch  
numpy  
opencv-python  
Pillow  
diffusers  
transformers  
accelerate  
controlnet-aux  
segment-anything

Depois execute:

pip install -r requirements.txt

---

## 📂 Estrutura de Pastas e Modelos

/
├── assets/                  # Imagens de entrada  
├── models/                  # Cache de modelos  
├── output/                  # Imagens geradas  
├── loras/                   # Arquivos LoRA  
│   └── nome_do_lora/  
├── reality2.safetensors     # Checkpoint SDXL  
├── sam_vit_h_4b8939.pth     # Checkpoint SAM (ViT-H)  
├── AIAssistant.py           # Classe principal  
├── Routines.py              # Lógica de execução  
├── image_to_image.py        # Entry point  
└── requirements.txt  

IMPORTANTE:
- Baixe o modelo oficial do SAM (sam_vit_h_4b8939.pth)
- Baixe um modelo SDXL (RealVisXL, Juggernaut, etc.)
- Renomeie para reality2.safetensors ou ajuste o caminho no código

---

## ⚙️ Configuração

Abra o arquivo image_to_image.py e ajuste os caminhos:

MODEL_PATH = "./seu_modelo_sdxl.safetensors"  
SAM_CHECKPOINT = "./sam_vit_h_4b8939.pth"  

Configure seus LoRAs:

MASTER_LORA = "./seus_loras/master.safetensors"

---

## 🚀 Como Usar

1. Coloque a imagem que deseja editar na pasta assets/
2. Execute o script principal:

python image_to_image.py

---

## 🧑‍💻 Fluxo Interativo no Terminal

- Digite o nome do arquivo (ex: minha_foto.png)
- Digite o Prompt e o Prompt Negativo
- Escolha o modo do IP-Adapter:
  - ENTER → Mantém o estilo da área
  - inv → Troca total (roupas/objetos)
  - caminho/para/imagem.jpg → Referência externa

---

## 🖼️ Interface Visual (OpenCV)

- Uma janela será aberta com a imagem
- Clique com o botão esquerdo nas áreas que deseja editar
- Pressione ESC para finalizar a seleção

---

## 🎛️ Ajuste de Parâmetros

O terminal solicitará:
- Denoising Strength
- CFG Scale
- Força dos Adapters

Pressione ENTER para usar os valores padrão.

A imagem final será salva na pasta output/.

---

## 🧠 Detalhes Técnicos

- Pipeline: StableDiffusionXLControlNetInpaintPipeline
- ControlNets: OpenPose e Depth Anything
- Otimização de memória:
  - gc.collect()
  - torch.cuda.empty_cache()
- Alternância inteligente entre SAM (RAM) e SDXL (VRAM)

---

## ⚠️ Solução de Problemas

Erro de Memória (OOM):
- O script usa max_split_size_mb:128
- Feche outros aplicativos que usam GPU
- Reduza a resolução da imagem

Arquivo não encontrado:
- Verifique se está na pasta assets/
- Confira nome e extensão (.jpg / .png)

---

## 📄 Licença

Projeto destinado a uso pessoal e educacional.

Verifique as licenças individuais dos modelos utilizados:
- Stable Diffusion XL
- Segment Anything Model (SAM)
- ControlNet
- LoRAs e checkpoints externos

---

Se este projeto te ajudou, considere deixar uma ⭐ no repositório!
