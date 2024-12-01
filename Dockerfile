# 使用官方 Python 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器
COPY . .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 运行 user_conversion.py 以生成模型文件
RUN python user_conversion.py

EXPOSE 8080

# 启动 Flask 应用
CMD ["python", "app.py"]
