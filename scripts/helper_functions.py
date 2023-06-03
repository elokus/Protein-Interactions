import pandas as pd
import numpy as np

def get_samples_UpDown_regulating(df, target_index, save_to_csv=False, return_dict=False, drop_col="None"):
    orig_df = df.drop(drop_col, axis=1)
    _df = orig_df.loc[target_index]
    df = _df.T
    print(_df)
    if isinstance(df, pd.Series):
        upreg = df.index[df > 0]
        dowreg = df.index[df < 0]
    else:
        upreg = df.index[df[target_index] > 0]
        dowreg = df.index[df[target_index] < 0]
    classes = []
    dict = {}
    for sample in orig_df.columns:
        if sample in upreg:
            classes.append(0)
            dict[sample] = 0
        elif sample in dowreg:
            classes.append(1)
            dict[sample] = 1
        else:
            print("not found ",sample)
    if isinstance(target_index, tuple):
        filename = f"output/CLS_{target_index[0]}_{target_index[1]}.csv"
    else:
        filename = f"output/CLS_{target_index}.csv"
    df = pd.DataFrame(classes)
    if save_to_csv:
        df.to_csv(filename)
    if return_dict:
        return dict
    return df


from scipy.special import betainc

def corrcoef(df):
    matrix = df.fillna(0).to_numpy()
    r = np.corrcoef(matrix)
    rf = r[np.triu_indices(r.shape[0], 1)]
    df = matrix.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = betainc(0.5 * df, 0.5, df / (df + ts))
    p = np.zeros(shape=r.shape)
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)] = p.T[np.tril_indices(p.shape[0], -1)]
    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])
    return r, p

def r_p_to_dataframe(r, p, df, target_gene, filter=False, min_pearson=0):
    _ind = df.index.get_loc(target_gene)
    df = pd.DataFrame({"Pval":p[_ind,:].T,"Pearson":r[_ind,:].T}, index=df.index).dropna(axis=1)
    if filter:
        df = df.where(df["Pval"]< 0.05, np.nan)
        df = df.where(abs(df["Pearson"]) > min_pearson).dropna(axis=0)
        return df
    return df

def cls_categorical(columns, categoric_series, replace_dict={}):
    arr = []
    for col in columns:
        arr.append(int(categoric_series.loc[col]))
    if len(replace_dict) > 0:
        for i in range(len(arr)):
            is_changed = False
            for key, value in replace_dict.items():
                if arr[i] == key and not is_changed:
                    arr[i] = value
                    is_changed = True
    return arr

def text_to_cls_file(categories: str, cls_numeric_cat: list, num_cat: int=2, num_samples: int=63, filename: str=None):
    lines = [f"{num_samples} {num_cat} 1", categories, cls_numeric_cat]
    if filename == None:
        filename = f"{categories}_CLS.cls"
    with open(filename, 'w') as f:
        for line in lines:
            if isinstance(line, list):
                text = " ".join(map(str,line))
                #print(text)
            else:
                text = line
            f.write(text)
            f.write("\n")
    return f"file saved as {filename}"

def cat_to_cls_file(columns, categoric_series, categories, replace_dict={}, filename=None, num_cat=2):
    cls_numeric_list = cls_categorical(columns, categoric_series, replace_dict)
    num_samples = len(cls_numeric_list)
    print(text_to_cls_file(categories, cls_numeric_list, num_cat, num_samples, filename))