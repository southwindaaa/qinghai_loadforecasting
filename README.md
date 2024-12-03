# 预测模型
# 组件目录
load_forecast_day96 -- 96点负荷预测模型

load_forecast_week -- 周度负荷预测模型

load_forecast_month -- 月度负荷预测模型

#### 具体组件可自行参考各组件目录下的README.md
# 环境配置
    python=3.9.14
    wheel==0.38.4
    setuptools==65.5.1
    pip==23.3.1
    torch==2.3.1
    torchvision==0.18.1
    torchaudio==2.3.1
    tqdm=4.65.0
    matplotlib==3.9.1
    transformers==4.34.1
    scipy==1.13.1
    pandas==2.2.2
    sklearn==1.5.2

#### 详细环境配置可参考requirements.txt

# 不同电网96点负荷预测
sh scripts/TimeLLM_SocialSmart_dlinear.sh
