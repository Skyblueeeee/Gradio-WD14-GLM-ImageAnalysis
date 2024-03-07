import argparse
import csv
import os

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

# from wd14 tagger
IMAGE_SIZE = 448

DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
FILES_ONNX = ["model.onnx"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]

class WD_TAG():
    def __init__(self) -> None:
        self.batch_size = 1
        self.thresh = 0.5
        self.caption_separator = ", "
        self.caption_extension = ".txt"
        self.model_dir = "models/wd/wd-v1-4-convnext-tagger-v2"
        self.repo_id = DEFAULT_WD14_TAGGER_REPO
        self.onnx = True #False
        self.character_threshold = None # threshold of confidence to add a tag for character category
        self.general_threshold = None   # threshold of confidence to add a tag for general category
        self.undesired_tags = ""       # comma-separated list of undesired tags to remove from the output
        self.append_tags=False       # Append captions instead of overwriting
        self.remove_underscore = None # replace underscores with spaces in the output tags

        if self.general_threshold is None:
            self.general_threshold = self.thresh
        if self.character_threshold is None:
            self.character_threshold = self.thresh

        self.model_exists()
        self.init_model()

    def model_exists(self):
        if not os.path.exists(self.model_dir):
            print(f"Downloading wd14 tagger model from hf_hub. id: {self.repo_id}")
            files = FILES
            if self.onnx:
                files += FILES_ONNX
            for file in files:
                hf_hub_download(self.repo_id, file, cache_dir=self.model_dir, force_download=True, force_filename=file)
            for file in SUB_DIR_FILES:
                hf_hub_download(
                    self.repo_id,
                    file,
                    subfolder=SUB_DIR,
                    cache_dir=os.path.join(self.model_dir, SUB_DIR),
                    force_download=True,
                    force_filename=file,
                )
        else:
            print("Using existing wd14 tagger model")
    
    def preprocess_image(self,image):
        # image = np.array(image)
        image = image[:, :, ::-1]  # RGB->BGR

        # pad to square
        size = max(image.shape[0:2])
        pad_x = size - image.shape[1]
        pad_y = size - image.shape[0]
        pad_l = pad_x // 2
        pad_t = pad_y // 2
        image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

        interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

        image = image.astype(np.float32)
        return image

    def collate_fn_remove_corrupted(self,batch):
        """Collate function that allows to remove corrupted examples in the
        dataloader. It expects that the dataloader returns 'None' when that occurs.
        The 'None's in the batch are removed.
        """
        # Filter out all the Nones (corrupted examples)
        batch = list(filter(lambda x: x is not None, batch))
        return batch

    def init_model(self):
        # 画像を読み込む
        if self.onnx:
            import onnx
            import onnxruntime as ort

            onnx_path = f"{self.model_dir}/model.onnx"
            print("Running wd14 tagger with onnx")
            print(f"loading onnx model: {onnx_path}")

            if not os.path.exists(onnx_path):
                raise Exception(
                    f"onnx model not found: {onnx_path}, please redownload the model with --force_download"
                    + " / onnxモデルが見つかりませんでした。--force_downloadで再ダウンロードしてください"
                )

            self.model = onnx.load(onnx_path)
            self.input_name = self.model.graph.input[0].name
            try:
                batch_size = self.model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
            except:
                batch_size = self.model.graph.input[0].type.tensor_type.shape.dim[0].dim_param

            if self.batch_size != batch_size and type(batch_size) != str:
                # some rebatch model may use 'N' as dynamic axes
                print(
                    f"Batch size {self.batch_size} doesn't match onnx model batch size {batch_size}, use model batch size {batch_size}"
                )
                self.batch_size = batch_size

            del self.model

            self.ort_sess = ort.InferenceSession(
                onnx_path,
                providers=["CUDAExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"],
            )
        else:
            from tensorflow.keras.models import load_model

            self.model = load_model(f"{self.model_dir}")

        # label_names = pd.read_csv("2022_0000_0899_6549/selected_tags.csv")
        # 依存ライブラリを増やしたくないので自力で読むよ

        with open(os.path.join(self.model_dir, CSV_FILE), "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            l = [row for row in reader]
            header = l[0]  # tag_id,name,category,count
            rows = l[1:]
        assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"unexpected csv format: {header}"

        self.general_tags = [row[1] for row in rows[1:] if row[2] == "0"]
        self.character_tags = [row[1] for row in rows[1:] if row[2] == "4"]
        self.tag_freq = {}

        self.stripped_caption_separator = self.caption_separator.strip()
        self.undesired_tags = set(self.undesired_tags.split(self.stripped_caption_separator))
    
    def run_batch(self,path_imgs):
        imgs = np.array([im for _, im in path_imgs])

        if self.onnx:
            if len(imgs) < self.batch_size:
                imgs = np.concatenate([imgs, np.zeros((self.batch_size - len(imgs), IMAGE_SIZE, IMAGE_SIZE, 3))], axis=0)
            probs = self.ort_sess.run(None, {self.input_name: imgs})[0]  # onnx output numpy
            probs = probs[: len(path_imgs)]
        else:
            probs = self.model(imgs, training=False)
            probs = probs.numpy()

        for (image_path, _), prob in zip(path_imgs, probs):
            # 最初の4つはratingなので無視する
            # # First 4 labels are actually ratings: pick one with argmax
            # ratings_names = label_names[:4]
            # rating_index = ratings_names["probs"].argmax()
            # found_rating = ratings_names[rating_index: rating_index + 1][["name", "probs"]]

            # それ以降はタグなのでconfidenceがthresholdより高いものを追加する
            # Everything else is tags: pick any where prediction confidence > threshold
            combined_tags = []
            general_tag_text = ""
            character_tag_text = ""
            for i, p in enumerate(prob[4:]):
                if i < len(self.general_tags) and p >= self.general_threshold:
                    tag_name = self.general_tags[i]
                    if self.remove_underscore and len(tag_name) > 3:  # ignore emoji tags like >_< and ^_^
                        tag_name = tag_name.replace("_", " ")

                    if tag_name not in self.undesired_tags:
                        self.tag_freq[tag_name] = self.tag_freq.get(tag_name, 0) + 1
                        general_tag_text += self.caption_separator + tag_name
                        combined_tags.append(tag_name)
                elif i >= len(self.general_tags) and p >= self.character_threshold:
                    tag_name = self.character_tags[i - len(self.general_tags)]
                    if self.remove_underscore and len(tag_name) > 3:
                        tag_name = tag_name.replace("_", " ")

                    if tag_name not in self.undesired_tags:
                        self.tag_freq[tag_name] = self.tag_freq.get(tag_name, 0) + 1
                        character_tag_text += self.caption_separator + tag_name
                        combined_tags.append(tag_name)

            # 先頭のカンマを取る
            if len(general_tag_text) > 0:
                general_tag_text = general_tag_text[len(self.caption_separator) :]
            if len(character_tag_text) > 0:
                character_tag_text = character_tag_text[len(self.caption_separator) :]

            caption_file = os.path.splitext(image_path)[0] + self.caption_extension

            tag_text = self.caption_separator.join(combined_tags)

            if self.append_tags:
                # Check if file exists
                if os.path.exists(caption_file):
                    with open(caption_file, "rt", encoding="utf-8") as f:
                        # Read file and remove new lines
                        existing_content = f.read().strip("\n")  # Remove newlines

                    # Split the content into tags and store them in a list
                    existing_tags = [tag.strip() for tag in existing_content.split(self.stripped_caption_separator) if tag.strip()]

                    # Check and remove repeating tags in tag_text
                    new_tags = [tag for tag in combined_tags if tag not in existing_tags]

                    # Create new tag_text
                    tag_text = self.caption_separator.join(existing_tags + new_tags)

            return tag_text
            # with open(caption_file, "wt", encoding="utf-8") as f:
            #     f.write(tag_text + "\n")
            #     if self.debug:
            #         print(f"\n{image_path}:\n  Character tags: {character_tag_text}\n  General tags: {general_tag_text}")

    def infer(self,image):
        # print(type(image))
        b_imgs = []
        # image = Image.open(img_path)
        # if image.mode != "RGB":
        #     image = image.convert("RGB")
        image = self.preprocess_image(image)

        b_imgs.append(("", image))

        if len(b_imgs) >= self.batch_size:
            b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
            tag = self.run_batch(b_imgs)
            b_imgs.clear()

        if len(b_imgs) > 0:
            b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
            tag = self.run_batch(b_imgs)
        return tag
    
if __name__ == "__main__":
    wti = WD_TAG()
    img_path = r"D:\shy_code\AIGC\lora-scripts-v1.4.1\data\王焕君画风\王焕君 (47).jpg"
    print(wti.infer(img_path))
