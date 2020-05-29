def score(df1,df2,df3):
    result = df1.append([df2, df3])
    result=result.reset_index().drop('index',axis=1)
    print(result)
    print("The best model is ",result[result['score']== result['score'].max()]["model"])
