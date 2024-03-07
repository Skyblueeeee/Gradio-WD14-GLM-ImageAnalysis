from transformers import AutoModel, AutoTokenizer  
from huggingface_hub import hf_hub_download
import re,os

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2
MODEL = "chatglm-6b"
DEFAULT_CHATGLM_REPO = "THUDM/chatglm-6b-int4"
INFO = """我给你关键词，你必须把它们组成一段话，需要优美和逻辑性，必须将句子翻译成中文。关键词："""

class CHATGLM():
    def __init__(self) -> None:
        self.model_dir = r"models\chatglm-6b"
        self.model_exists()
        self.init_model()

    def model_exists(self):
        if not os.path.exists(self.model_dir):
            print(f"Downloading chatglm model from hf_hub")
            hf_hub_download(DEFAULT_CHATGLM_REPO,MODEL, cache_dir=self.model_dir, force_download=True, force_filename=MODEL)

        else:
            print("Using existing chatglm-6b model")

    def init_model(self):
        # 加载预训练的tokenizer和model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)  
        model = AutoModel.from_pretrained(self.model_dir, trust_remote_code=True).half().cuda(0)
        self.model = model.eval()  # 设置model为评估模式

    def post_res(self,input_string):
        chinese_characters_and_punctuation = re.findall(r'[\u4e00-\u9fff\uf900-\ufaff\uff01-\uffef]+', input_string)
        return ''.join(chinese_characters_and_punctuation)

    def predict(self,input, max_length=2048, top_p=0.9, temperature=0.95, history=None):
        inputs = INFO + input
        first_ans = ('','')
        second_ans = ('','')
        if history is None:
            history = []
        for response, history in self.model.stream_chat(self.tokenizer, inputs, max_length=max_length, top_p=top_p,
                        temperature=temperature, history=(history if len(history) <= 4 else [first_ans] + [second_ans] + history[-3:])):
            if len(history) <= 1 :
                first_ans = history[0]
            if len(history) <= 2 and len(history) > 1 :
                second_ans = history[1]
        return input +"\n\n" + response
if __name__ == "__main__":
    glm = CHATGLM()
    while True:
        text = input("请输入关键词:")
        rep = glm.predict(text)
        print(glm.post_res(rep))
