import os
from huggingface_hub import snapshot_download

def download_models():
    # 使用国内的高速镜像站，免翻墙且速度极快
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 对应的 Hugging Face 模型 ID
    models = [
        "BAAI/bge-base-zh-v1.5",
        "BAAI/bge-reranker-base"
    ]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for model_id in models:
        model_name = model_id.split('/')[-1]
        target_path = os.path.join(current_dir, model_name)
        
        print(f"正在从镜像站下载: {model_id} -> {target_path}")
        try:
            # huggingface_hub 的 snapshot_download 不需要任何 PyTorch 依赖
            snapshot_download(
                repo_id=model_id,
                local_dir=target_path,
                local_dir_use_symlinks=False
            )
            print(f" 成功完成: {model_name} 下载成功！\n")
        except Exception as e:
            print(f" ❌ 下载失败: {model_id}，错误原因: {e}\n")

if __name__ == "__main__":
    download_models()