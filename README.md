# RecSys Challenge 2022 - 48th Place Solution

This repository contains the code for the 48th place solution in the [RecSys Challenge 2022](https://www.recsyschallenge.com/2022/).

## Dataset
[Dressipi 1M Fashion Sessions Dataset](https://dressipi.com/downloads/recsys-datasets/)


## Solution Overview
1. Retrieval Strategies.
2. Feature Engineering.
3. Ranking Model - Gradient Boosting Tree.


## Candidate Retrieval Strategies
In this stage, we focus on retrieving as many candidate items as possible to cover more positive samples, as the hit number of candidate items sets the upper bound of the accuracy limitation. 

Also, we will retrieve items from a recommended item set defined by RecSys Challenge, which consists of 4990 items.

- Popularity:
    - Top *K* frequently clicked items in the last month.
    - Top *K* frequently purchased items in the last month.
- Item2Item Similarity:
    - Top *K* similar items for each item in the current session.
- Session2Item Similarity: 
    - Top *K* similar items based on session embeddings.
    - Session embedding is the average of item embeddings.


## Features
- Similarity - item2item
  - Last N-th item vs. candidate item
- Similarity - session2item
  - Average item embedding of last N items in each session vs. candidate item
- Clicked / Purchased Count
  - Clicked count of candidate item
- Purchased count of candidate item
  - Same Category / Attribute Count
- Last N-th item vs. candidate item
  - Session vs. candidate item
- Category & Attribute
  - Candidate item


## Model
Fashion is a fast-changing field, where seasons and current fashion trends significantly influence user interests. 
Therefore, focusing on recent transactions provides more valuable insights into user behavior than long-term historical data. 
In this competition, we used transaction data from the last three months for training and validation, as shown in Fig. 1.
We implemented the LightGBM Ranker as our primary model due to its proven stability and strong performance in similar competitions.
![trainval](https://github.com/user-attachments/assets/ec5485e0-df92-489b-b0f5-28750a062145)










