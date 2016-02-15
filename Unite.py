import pandas as pd 


minmax = [(1,2),(3,4),(5,6),(7,8)]


def modified_responses(x, minmax):
    #minmax here is a tuple, e.g. (1,2)
    buffer = 0
    if min(minmax)==1: buffer = x
    else: buffer = x+min(minmax)-2
    if buffer > 8: return 8
    else: return buffer

def main():

    path =  ('/home/reinhold/data/ML/Prudential/intermediate_data/',
             '/home/reinhold/data/ML/Prudential/output_data/')

    outfilename = path[1] + "test_Prudential_predicted_labels_BDT.csv"

    #for minmax_ in minmax:
    #    for i in range(1,5):
    #        print(i, minmax_, modified_responses(i,minmax_))

    df,df_BDT = [],[]

    for minmax_ in minmax:
        df[len(df):] = [pd.read_csv(path[0] + "test_Prudential_predicted_labels_pred_resp%d-%d.csv" % (minmax_[0], minmax_[1]), header=0)]
        df_BDT[len(df_BDT):] = [pd.read_csv(path[1] + "test_Prudential_predicted_labels_BDT_pred_resp%d-%d.csv" % (minmax_[0], minmax_[1]), header=0)]


    for (counter, (i,j, minmax_)) in enumerate(zip(df,df_BDT, minmax)):

        print(counter, i.shape, j.shape, minmax_)
        i = i.join(j, how='inner', rsuffix='_BDT')
        i = i[(i['Response_pred']==minmax_[0]) |(i['Response_pred']==minmax_[1])] #for the test datasets this is already done
        #print(i.columns)
        i['Response_BDT'] = i['Response_BDT'].apply(lambda x: modified_responses(x, minmax_))
        #print(i.shape)
        if counter==0: df_out = i
        elif counter: df_out = df_out.append(i) #Note:Unlike list.append method, which appends to the original list and returns nothing, append here does not modify df1 and returns its copy with df2 appended. http://pandas.pydata.org/pandas-docs/version/0.13.1/merging.html#concatenating-using-append

    #print(df_out)
    print(df_out.shape)
    
    preds_out = pd.DataFrame({"Id": df_out['Id'].values, "Response": df_out['Response_BDT']})
    preds_out = preds_out.set_index('Id')
    preds_out.to_csv(outfilename)
    print("file created: ", outfilename)


if __name__ == "__main__":
    main()
