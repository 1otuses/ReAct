import json
import os
import gymnasium as gym
import numpy as np
import re
import string
from collections import Counter

DATA_DIR = "data"
HOTPOTQA_SPLIT_FILE = {
    "train": "hotpot_train_v1.1_simplified.json",
    "dev": "hotpot_dev_v1_simplified.json",
    "test": "hotpot_test_v1_simplified.json",
}

FEVER_SPLIT_FILE = {
    "train": "train.jsonl",
    "dev": "paper_dev.jsonl",
}


class HistoryWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_format, prompt=None):
        super().__init__(env)
        assert obs_format in ["obs", "history"]
        if obs_format == "history":
            assert hasattr(self.env, "traj")
        self.obs_format = obs_format
        self.prompt = prompt if prompt is not None else ""

    def observation(self, obs):
        if self.obs_format == "obs":
            return obs
        elif self.obs_format == "history":
            observation = self.env.traj["observations"][0] + "\n"
            for i, (o, a) in enumerate(zip(self.env.traj["observations"][1:], self.env.traj["actions"]), 1):
                observation += f"Action {i}: {a}\nObservation {i}: {o}\n\n"
            return self.prompt + observation


def normalize_answer(s):  # 对文本答案标准归一化
    def remove_articles(text):  # 去除冠词(a、an、the)
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):  # 空格规范化
        return " ".join(text.split())

    def remove_punc(text):  # 去除标点符号
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):  # 统一小写
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):  # prediction:预测答案,ground_truth:正确答案
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    # 对于答案属于['yes', 'no', 'noanswer']类型的问题,答案匹配需要完全一致,如果不同直接返回0奖励

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)  # Counter用于计算共同单词总数
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall  # F1分数,预测正确率,召回率


class HotPotQAWrapper(gym.Wrapper):
    def __init__(self, env, split):
        super().__init__(env)
        self.steps = 0  # current number of steps
        self.answer = None  # current answer from the agent
        data_file = f"{DATA_DIR}/{HOTPOTQA_SPLIT_FILE[split]}"  # 加载HotPotQA数据集
        self.data = json.load(open(data_file))
        self.data = [(d['question'], d['answer']) for d in self.data]  # 从数据中提取问题和答案
        # data = [(question1, answer1), (question2, answer2), ...]
        self.data_idx = 0
        self.split = split  # "train", "dev", "test"

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        self.steps = 0
        self.answer = None

        self.env.reset(seed=seed, return_info=True, options=options)
        self.data_idx = np.random.randint(len(self.data)) if idx is None else idx
        observation = f"Question: {self.data[self.data_idx][0]}"
        # self.env.reset(seed=seed, return_info=return_info, options=options)
        # try:
        #     self.env.step('')
        # except:
        #     pass
        # self.env.reset(seed=seed, return_info=return_info, options=options)
        # 两次reset和一次空step操作: 确保底层wikienv内部状态恢复到了初始状态
        # self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx  # 随机采样一个问题
        # observation = f"Question: {self.data[self.data_idx][0]}"
        info = self._get_info()
        return observation, info

    def _get_info(self):  # 获取信息
        return {
            "steps": self.steps,
            "answer": self.answer,
            "question": self.data[self.data_idx][0],
            "hotpot_split": self.split
        }

    def get_reward(self, info):  # 二分类奖励
        if info['answer'] is not None:
            pred = normalize_answer(info['answer'])  # 对预测答案标准化
            gt = normalize_answer(self.data[self.data_idx][1])  # 对真实答案标准化
            score = (pred == gt)
            return int(score)
        return 0

    def get_metrics(self, info):  #
        if info['answer'] is not None:
            pred = normalize_answer(info['answer'])
            gt = normalize_answer(self.data[self.data_idx][1])
            em = (pred == gt)
            f1 = f1_score(pred, gt)[0]
            return {'reward': em, 'em': em, 'f1': f1}
        return {'reward': 0, 'em': 0, 'f1': 0}

    def step(self, action):
        # TODO: first step obs does not have question.
        # obs, _, _, done, info = self.env.step(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated  # 兼容原逻辑中的done判断

        self.steps += 1
        if "answer" in info:
            self.answer = info["answer"]
        reward = self.get_reward(info)
        if done:
            obs = f"Episode finished, reward = {reward}\n"
            info.update({"gt_answer": self.data[self.data_idx][1],  # 真实答案
                         "question_idx": self.data_idx,
                         "steps": self.steps,
                         "answer": self.answer
                         })
            info.update(self.get_metrics(info))
        return obs, reward, terminated, truncated, info

    def __len__(self):
        return len(self.data)


class FeverWrapper(gym.Wrapper):
    def __init__(self, env, split):
        super().__init__(env)

        self.steps = 0  # current number of steps
        self.answer = None  # current answer from the agent
        data_path = f"./data/{FEVER_SPLIT_FILE[split]}"
        with open(data_path, "r") as json_file:
            json_list = list(json_file)

        data = []
        for json_str in json_list:
            json_str = json.loads(json_str)
            label = json_str["label"]
            claim = json_str["claim"]
            data.append((claim, label))

        self.data = data
        self.data_idx = 0
        self.split = split

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        self.env.reset(seed=seed, return_info=return_info, options=options)
        try:
            self.env.step('')
        except:
            pass
        self.env.reset(seed=seed, return_info=return_info, options=options)
        self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
        observation = f"Claim: {self.data[self.data_idx][0]}"
        info = self._get_info()
        return observation, info

    def _get_info(self):
        return {
            "steps": self.steps,
            "answer": self.answer,
            "question": self.data[self.data_idx][0],
            "fever_split": self.split
        }

    def get_reward(self, info):
        if info['answer'] is not None:
            label = normalize_answer(self.data[self.data_idx][1])
            pred = normalize_answer(info['answer'])
            if label == pred:
                return 1
        return 0

    def step(self, action):
        # TODO: first step obs does not have question.
        # obs, _, _, done, info = self.env.step(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        reward = self.get_reward(info)
        if done:
            obs = f"Episode finished, reward = {reward}\n"
            info.update({"gt_answer": self.data[self.data_idx][1],
                         "question_idx": self.data_idx
                         })
            info.update({'em': reward, 'reward': reward, 'f1': reward})
        return obs, reward, terminated, truncated, info

    def __len__(self):
        return len(self.data)


class LoggingWrapper(gym.Wrapper):
    def __init__(self, env, folder="trajs", file_id=None):  # 记录运行轨迹Trajectory自动存于文件trajs
        super().__init__(env)
        self.trajs = []  # 记录所有轨迹
        self.traj = {"observations": [], "actions": []}  # 记录当前轨迹每一步的obs,act
        self.folder = folder
        self.file_id = np.random.randint(0, 10000000) if file_id is None else file_id
        self.file_path = f"{self.folder}/{self.file_id}.json"
        os.makedirs("trajs", exist_ok=True)

    def __len__(self):
        return len(self.env.data)

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        obs, info = self.env.reset(seed=seed, return_info=True, options=options, idx=idx)
        # observation = output[0] if return_info else output
        self.traj = {"observations": [obs], "actions": []}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        # obs, reward, _, done, info = self.env.step(action)
        self.traj["observations"].append(obs)
        self.traj["actions"].append(action)
        if done:
            self.traj.update(info)
        return obs, reward, terminated, truncated, info

    def update_record(self):
        if len(self.traj["actions"]) > 0:
            self.trajs.append(self.traj)
            self.traj = {"observations": [], "actions": []}

    def write(self):
        self.update_record()
        with open(self.file_path, "w") as f:
            json.dump(self.trajs, f)
            print(f"Saved trajs to trajs/{self.file_id}.json")

    def close(self):
        self.write()