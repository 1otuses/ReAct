import ast
import json
import time
import gymnasium as gym
import requests
from bs4 import BeautifulSoup


# import wikipedia

def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


class textSpace(gym.spaces.Space):  # 任意一个文本都可以作为一个动作空间
    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return isinstance(x, str)


class WikiEnv(gym.Env):

    def __init__(self):
        """
      Initialize the environment.
    """
        super().__init__()
        self.page = None  # current Wikipedia page 当前页面
        self.obs = None  # current observation 环境返回给Agent的文本
        self.lookup_keyword = None  # current lookup keyword  当前页面中的关键字
        self.lookup_list = None  # list of paragraphs containing current lookup keyword 所有包含关键字的句子
        self.lookup_cnt = None  # current lookup index 当前包含关键字的句子的下标
        self.steps = 0  # current number of steps
        self.answer = None  # current answer from the agent 答案:finish[answer]
        self.observation_space = self.action_space = textSpace()
        self.search_time = 0  # 搜索耗时
        self.num_searches = 0  # 搜索次数

    def _get_obs(self):
        return self.obs

    def _get_info(self):
        return {"steps": self.steps, "answer": self.answer}

    def reset(self, seed=None, return_info=False, options=None):  # 重置
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                    "finish[].\n")
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def construct_lookup_list(self, keyword):  # 从当前页面中提取出所有包含关键字的句子
        # find all paragraphs
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]  # 分离段落
        # paragraphs = ["paragraph1","paragraph2","paragraph3"]

        # find all sentence
        sentences = []
        for p in paragraphs:  # 分离句子
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        # sentences = ["sentence1","sentence2","sentence3"]

        parts = sentences
        parts = [p for p in parts if keyword.lower() in p.lower()]  # 句子中匹配关键词
        return parts

    @staticmethod
    def get_page_obs(page):  # 从当前页面中选取前5句作为观察
        # find all paragraphs
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return ' '.join(sentences[:5])

        # ps = page.split("\n")
        # ret = ps[0]
        # for i in range(1, len(ps)):
        #   if len((ret + ps[i]).split(" ")) <= 50:
        #     ret += ps[i]
        #   else:
        #     break
        # return ret

    def search_step(self, entity):  # 进行search[entity]步骤
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        old_time = time.time()
        response_text = requests.get(search_url).text
        self.search_time += time.time() - old_time
        self.num_searches += 1
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:  # mismatch
            self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."  # 未搜索到信息,生成相似的5段内容
        else:
            page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
            # 读取页面,从<p>和<ul>中
            if any("may refer to:" in p for p in page):
                self.search_step("[" + entity + "]")  # 递归调用search[entity],处理歧义文本
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:  # 过滤过短的段落
                        self.page += clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"  # 段落回车结尾
                self.obs = self.get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None  # 重置状态

    def step(self, action):
        reward = 0
        terminated = False  # 任务完成的终止状态
        truncated = False  # 超时等截断状态（暂设为False）
        action = action.strip()
        if self.answer is not None:  # already finished answer不为空(有答案)
            terminated = True  # 任务完成
            return self.obs, reward, terminated, truncated, self._get_info()

        if action.startswith("search[") and action.endswith("]"):  # 动作属于search[entity]
            entity = action[len("search["):-1]
            # entity_ = entity.replace(" ", "_")
            # search_url = f"https://en.wikipedia.org/wiki/{entity_}"
            self.search_step(entity)  # 更新page、obs
        elif action.startswith("lookup[") and action.endswith("]"):  # 动作属于lookup[keyword],类似网页中Ctrl+F功能
            keyword = action[len("lookup["):-1]
            if self.lookup_keyword != keyword:  # reset lookup
                self.lookup_keyword = keyword  # 更新关键词
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                self.obs = "No more results.\n"
            else:
                self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
                self.lookup_cnt += 1
        elif action.startswith("finish[") and action.endswith("]"):  # 动作属于finish[answer]
            answer = action[len("finish["):-1]  # finish[]括号中的内容
            self.answer = answer
            terminated = True
            self.obs = f"Episode finished, reward = {reward}\n"
        elif action.startswith("think[") and action.endswith("]"):  # 动作属于think[obs]
            self.obs = "Nice thought."
        else:
            self.obs = "Invalid action: {}".format(action)  # 无效动作指令

        self.steps += 1

        return self.obs, reward, terminated, truncated, self._get_info()

    def get_time_info(self):
        speed = self.search_time / self.num_searches if self.num_searches else 0
        return {
            "call_speed": speed,
            "call_time": self.search_time,
            "num_calls": self.num_searches,
        }
