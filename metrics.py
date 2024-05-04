import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.api as sm
## load dataset
sample_posts = pd.read_csv("C://Users/vvalmeekam/WorkSpace/projects/p12/input/reddit/askscience_data.csv",header=0)
## compute comment counts for each post
def count_comments(df):
    df["comment_count"] = 0
    for i in range(sample_posts.shape[0]):
        body = sample_posts["body"][i]
        if type(body) != str:
            body = ""
        body_list = body.split("\n")
        body_list_clean = list()
        for k in range(len(body_list)):
            if body_list[k] != "":
                body_list_clean.append(body_list[k])
        df.at[i,"comment_count"] = len(body_list_clean)
    return df
# distribution of score, comment_counts, upvote_ratio as a metric for success
def get_score_dist(df):
    want_q = np.arange(0,1.1,0.1)
    df = df["score"].quantile(q=want_q)
    return df.to_frame()
def get_cc_dist(df):
    want_q = np.arange(0,1.1,0.1)
    df = df["comment_count"].quantile(q=want_q)
    return df.to_frame()
def get_upvote_ratio_dist(df):
    want_q = np.arange(0,1.1,0.1)
    df = df["upvote_ratio"].quantile(q=want_q)
    return df.to_frame()
## correlation of variables
def corr_upvote_ratio(df):
    corr, pval = spearmanr(df["score"], df["upvote_ratio"])
    print(corr, pval)
def corr_comment_count(df):
    corr, pval = spearmanr(df["score"], df["comment_count"])
    print(corr, pval)
    corr, pval = spearmanr(df["upvote_ratio"], df["comment_count"])
    print(corr, pval)
def do_linreg(df):
    df["tag"] = df["tag"].astype("category")
    df["tag"] = df["tag"].cat.codes
    x = df[["upvote_ratio"]].to_numpy()
    y = df["comment_count"].to_numpy()
    model = sm.GLS(y, x).fit()
    print(model.summary(alpha = 0.05))
score_dist = get_score_dist(sample_posts)
upvote_dist = get_upvote_ratio_dist(sample_posts)
corr_upvote_ratio(sample_posts)
df = count_comments(sample_posts)
corr_comment_count(df)
count_dist = get_cc_dist(df)
do_linreg(df)
## other exploratory analyses
## chose 60% percentile of score value as cutoff for success
#success_thresh = score_dist.iloc[6]["score"]
# sb.scatterplot(data=df, x="score", y="comment_count")
# plt.show()
### what is the association between score and upvote_ratio within each tag?
# for mtag in df["tag"].unique():
#     print(mtag)
#     mdf = df[df["tag"] == mtag]
#     print(mdf["score"].median(), mdf.shape[0])
#     corr, pvalue = spearmanr(mdf["score"], mdf["upvote_ratio"])
#     print(mtag, corr,pvalue)
# g = sb.FacetGrid(df, col="tag")
# g.map(sb.scatterplot, "score", "upvote_ratio")
# plt.show()
# df = sample_posts.groupby(["tag"])["score"].mean()
# print(df.sort_values())
# df = sample_posts.groupby(["tag"])["score"].size()
# print(df.sort_values())