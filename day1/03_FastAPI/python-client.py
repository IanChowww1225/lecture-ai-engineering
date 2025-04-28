# python_client.py
# このコードは、ngrokで公開されたAPIにアクセスするPythonクライアントの例です

import requests
import json
import time

# API 配置
API_URL = "http://localhost:8000"

def generate_text(prompt: str, max_length: int = 1000, temperature: float = 0.7):
    """生成文本"""
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {str(e)}")
        return None

def check_health():
    """检查服务健康状态"""
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"健康检查失败: {str(e)}")
        return None

def main():
    """主函数"""
    # 检查服务健康状态
    health = check_health()
    if health:
        print("服务状态:", health)
    
    # 测试生成
    prompt = "请介绍一下人工智能的发展历史。"
    result = generate_text(prompt)
    
    if result:
        print("\n生成的文本:")
        print(result["response"])
        print("\n模型信息:")
        print(json.dumps(result["model_info"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

class LLMClient:
    """LLM API クライアントクラス"""
    
    def __init__(self, api_url):
        """
        初期化
        
        Args:
            api_url (str): API のベース URL（ngrok URL）
        """
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self):
        """
        ヘルスチェック
        
        Returns:
            dict: ヘルスチェック結果
        """
        response = self.session.get(f"{self.api_url}/health")
        return response.json()
    
    def generate(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True):
        """
        テキスト生成
        
        Args:
            prompt (str): プロンプト文字列
            max_new_tokens (int, optional): 生成する最大トークン数
            temperature (float, optional): 温度パラメータ
            top_p (float, optional): top-p サンプリングのパラメータ
            do_sample (bool, optional): サンプリングを行うかどうか
        
        Returns:
            dict: 生成結果
        """
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample
        }
        
        start_time = time.time()
        response = self.session.post(
            f"{self.api_url}/generate",
            json=payload
        )
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            result["total_request_time"] = total_time
            return result
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")

# 使用例
if __name__ == "__main__":
    # ngrok URLを設定（実際のURLに置き換えてください）
    NGROK_URL = "https://your-ngrok-url.ngrok.url"
    
    # クライアントの初期化
    client = LLMClient(NGROK_URL)
    
    # ヘルスチェック
    print("Health check:")
    print(client.health_check())
    print()
    
    # 単一の質問
    print("Simple question:")
    result = client.generate([
        {"prompt": "AIについて100文字で教えてください"}
    ])
    print(f"Response: {result['generated_text']}")
    print(f"Model processing time: {result['response_time']:.2f}s")
    print(f"Total request time: {result['total_request_time']:.2f}s")    