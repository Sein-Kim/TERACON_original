## Task Relation-aware Continual User Representation Learning

The official source code for [**Task Relation-aware Continual User Representation Learning**](https://arxiv.org/abs/2306.01792) paper, accepted at KDD 2023.

## Abstract
User modeling, which learns to represent users into a low-dimensional representation space based on their past behaviors, got a surge of interest from the industry for providing personalized services to users. Previous efforts in user modeling mainly focus on learning a task-specific user representation that is designed for a single task. However, since learning task-specific user representations for every task is infeasible, recent studies introduce the concept of universal user representation, which is a more generalized representation of a user that is relevant to a variety of tasks. Despite their effectiveness, existing approaches for learning universal user representations are impractical in real-world applications due to the data requirement, catastrophic forgetting and the limited learning capability for continually added tasks. In this paper, we propose a novel continual user representation learning method, called TERACON, whose learning capability is not limited as the number of learned tasks increases while capturing the relationship between the tasks. The main idea is to introduce an embedding for each task, i.e., task embedding, which is utilized to generate task-specific soft masks that not only allow the entire model parameters to be updated until the end of training sequence, but also facilitate the relationship between the tasks to be captured. Moreover, we introduce a novel knowledge retention module with pseudo-labeling strategy that successfully alleviates the long-standing problem of continual learning, i.e., catastrophic forgetting. Extensive experiments on public and proprietary real-world datasets demonstrate the superiority and practicality of TERACON. 

## Dataset

You can download the datasets from this url from [CONURE](https://arxiv.org/abs/2009.13724)<br>

TTL: https://drive.google.com/file/d/1imhHUsivh6oMEtEW-RwVc4OsDqn-xOaP/view<br>

MovieLens: https://grouplens.org/datasets/movielens/25m/

---
The datapath_index "Data/session/index.csv"
be automatically generated when running task1.
More specifically, if you run the dataset for task1, the data_loader generate index.csv for all items in task1.

## How to Run
First run the task 1
train_task1.py
to get the model which train task1
<br>

~~~
python train_task1.py --epochs 5 --lr 0.001 --batch 32
~~~

---
Then run squentially,
1. train_teracon_t2.py
2. train_teracon_t3.py
3. train_teracon_t4.py
4. train_teracon_t5.py
5. train_teracon_t6.py
<br>
E.g.,<br>

~~~
python train_teracon_t2.py --lr 0.0001 --smax 50 --batch 1024
~~~

<br>

~~~
python train_teracon_t3.py --lr 0.0001 --smax 50 --batch 1024
~~~

## Backbone network Code
The data_loder code and basic backbone network are refered to<br>

https://github.com/yuangh-x/2022-NIPS-Tenrec

https://github.com/syiswell/NextItNet-Pytorch

---
This code is fitted to TTL data set, if you run the ML datasets please change the paths of data

### Cite (Bibtex)
- Please refer the following paer, if you find TERACON useful in your research:
  - Kim, Sein and Lee, Namkyeong and Kim, Donghyun and Yang, Minchul and Park, Chanyoung. "Task Relation-aware Continual User Representation Learning." KDD 2023.
  - Bibtex
```
@article{kim2023task,
  title={Task Relation-aware Continual User Representation Learning},
  author={Kim, Sein and Lee, Namkyeong and Kim, Donghyun and Yang, Minchul and Park, Chanyoung},
  journal={arXiv preprint arXiv:2306.01792},
  year={2023}
}
```
