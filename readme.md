# Image Upscaler Using Deep Learning (AI) 📸

└ Fork do projeto [mehrdad-dev/Image-Upscaler-Deep-Learning](https://github.com/mehrdad-dev/Image-Upscaler-Deep-Learning)

# Models
1. EDSR - [Paper](https://arxiv.org/pdf/1707.02921.pdf)
2. LapSRN - [Paper](https://arxiv.org/pdf/1710.01992.pdf)
3. FSRCNN -  [Paper](https://arxiv.org/pdf/1608.00367.pdf)
4. ESPCN - [Paper](https://arxiv.org/pdf/1609.05158.pdf)   
5. Real-ESRGAN - [Paper](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.pdf)

---

## Instalação

### Clonar o projeto
    
    git clone https://github.com/fahadkalil/image_upscaler.git    
    cd image_upscaler

### Criar e ativar o virtualenv para o projeto

- Windows    
      
      python -m venv .venv
      .venv/Scripts/activate.bat

- Linux / MacOS
  
      cd image_upscaler
      python -m venv .venv
      source .venv/bin/activate

### Instalar dependências
    
    pip install -r requirements.txt

### Para rodar (confira a saída do comando)

    streamlit run streamlit_app.py
