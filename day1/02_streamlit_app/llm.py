# llm.py
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import time
from config import MODEL_NAME, HUGGINGFACE_TOKEN, TEMPERATURE, MAX_LENGTH
from huggingface_hub import login

# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        # アクセストークンを保存
        hf_token = st.secrets["huggingface"]["token"]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None

def generate_response(pipe, user_question):
    """LLMを使用して質問に対する回答を生成する"""
    if pipe is None:
        return "モデルがロードされていないため、回答を生成できません。", 0

    try:
        start_time = time.time()
        messages = [
            {"role": "user", "content": user_question},
        ]
        # max_new_tokensを調整可能にする（例）
        outputs = pipe(messages, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)

        # Gemmaの出力形式に合わせて調整が必要な場合がある
        # 最後のassistantのメッセージを取得
        assistant_response = ""
        if outputs and isinstance(outputs, list) and outputs[0].get("generated_text"):
           if isinstance(outputs[0]["generated_text"], list) and len(outputs[0]["generated_text"]) > 0:
               # messages形式の場合
               last_message = outputs[0]["generated_text"][-1]
               if last_message.get("role") == "assistant":
                   assistant_response = last_message.get("content", "").strip()
           elif isinstance(outputs[0]["generated_text"], str):
               # 単純な文字列の場合（古いtransformers？） - プロンプト部分を除く処理が必要かも
               # この部分はモデルやtransformersのバージョンによって調整が必要
               full_text = outputs[0]["generated_text"]
               # 簡単な方法：ユーザーの質問以降の部分を取得
               prompt_end = user_question
               response_start_index = full_text.find(prompt_end) + len(prompt_end)
               # 応答部分のみを抽出（より堅牢な方法が必要な場合あり）
               possible_response = full_text[response_start_index:].strip()
               # 特定の開始トークンを探すなど、モデルに合わせた調整
               if "<start_of_turn>model" in possible_response:
                    assistant_response = possible_response.split("<start_of_turn>model\n")[-1].strip()
               else:
                    assistant_response = possible_response # フォールバック

        if not assistant_response:
             # 上記で見つからない場合のフォールバックやデバッグ
             print("Warning: Could not extract assistant response. Full output:", outputs)
             assistant_response = "回答の抽出に失敗しました。"


        end_time = time.time()
        response_time = end_time - start_time
        print(f"Generated response in {response_time:.2f}s") # デバッグ用
        return assistant_response, response_time

    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        # エラーの詳細をログに出力
        import traceback
        traceback.print_exc()
        return f"エラーが発生しました: {str(e)}", 0

class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """加载模型和分词器"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                token=HUGGINGFACE_TOKEN
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                token=HUGGINGFACE_TOKEN,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            raise Exception(f"模型加载失败: {str(e)}")

    def generate_response(self, prompt):
        """生成回复"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            raise Exception(f"生成回复失败: {str(e)}")

    def get_model_info(self):
        """获取模型信息"""
        return {
            "name": MODEL_NAME,
            "temperature": TEMPERATURE,
            "max_length": MAX_LENGTH
        }