import os, pandas as pd
import json, numpy as np
from PIL import Image, ImageFile
import pickle

class DataReady:
    def __init__(self, csv_file="book_data_final.csv"):
        self.df = pd.read_csv(csv_file, encoding='ISO-8859-1')
        self.label_dict = None
        self.json_file="img_labelscore_dict.json"
        
        # when get objects with scores in get_obj_score()
        self.cut_length = 1
        
        self.inputSize_wh = 32
        self.imgarray_filename = '../book_dataset{}/imgarray.pickle'.format(self.inputSize_wh)

    def get_df(self): return self.df
    
    def drop_reset(self):
        self.df.dropna(inplace=True,subset=["review", "label"])
        self.df.reset_index(drop=True,inplace=True)
        for i in range(len(self.df["number"])):
            if i != self.df["number"][i]:
                self.df["number"][i] = i
    
    # img flow 
    def get_imgarray(self, inputSize_wh=None):
        if inputSize_wh is None:
            imgarray_filename = self.imgarray_filename
        else:
            self.inputSize_wh = inputSize_wh
            imgarray_filename = '../book_dataset{}/imgarray.pickle'.format(self.inputSize_wh)
        print(f"load '{imgarray_filename}'")
        if os.path.isfile(imgarray_filename):
            print("파일이 존재합니다.")
            with open(imgarray_filename, 'rb') as fp:
                num2resizedImg = pickle.load(fp)
            print("불러온 data #: {}".format(len(num2resizedImg)))
        else:
            print("파일을 생성합니다")
            
            # 시스템 내의 이미지 디렉토리의 리스트
            imagedirs = os.listdir('book_dataset')
            imagedirs.remove(".ipynb_checkpoints")

            # number를 key값, image를 value값으로 하는 dict 파일(num2img) 생성
            num2img = {}
            for num in self.df['number'][:]:
                img = Image.open('book_dataset/' + self.df['id'][num])
                num2img[num] = img

            # input size 지정
            InputSize = (self.inputSize_wh, self.inputSize_wh)

            ImageFile.LOAD_TRUNCATED_IMAGES = True

            # image를 array로 변환해 저장
            num2resizedImg = {}
            for key,value in num2img.items():
                if value != None:
                    num2resizedImg[key] = np.array(value.resize(InputSize),dtype=np.float)/255.0
                else:
                    num2resizedImg[key] = None

            # 채널 1개인 이미지를 3개로 변환
            for i in range(len(number)):
                if num2resizedImg[i].shape != (self.inputSize_wh, self.inputSize_wh, 3):
                    # normalize
                    norm = cv2.normalize(num2resizedImg[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    # convert to 3 channel
                    num2resizedImg[i] = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
                    
            with open(imgarray_filename, 'wb') as fp:
                pickle.dump(num2resizedImg, fp)
            
        return num2resizedImg
    
    # img object flow
    def get_img_label_score(self):
        if self.json_file not in os.listdir():
            yolo_model = YoloPredict(self.json_file, self.df)
            yolo_model.make_json()
        
        with open(self.json_file,"r") as f:
            self.label_dict=json.load(f)
            
        self.df["objects"]=""
        self.df["scores"]=""
        for ind, d in enumerate(self.df.id):
            self.df["objects"].iloc[ind]=self.label_dict[d][0]
            self.df["scores"].iloc[ind]=self.label_dict[d][1]
            
    def get_obj_score(self):
        obj_list, score_list = [],[]
        for t in self.df[["objects","scores"]].values:
            obj, scr = t[0], t[1]
            obj_list.append(obj[:self.cut_length])
            score_list.append(scr[:self.cut_length])
        return np.array(obj_list), np.array(score_list)