import pandas as pd
import scipy.stats


class ShapiroAggregator:
    def __init__(self, path, name="drug_viability"):
        self.name = name
        self.df = self.read_file(path)
        self.df_orig = self.df.copy()
        self.df_classified = self.df.set_index(["expression", "substance", "dose", "timepoint"])
        self.sample_classes = self.df_classified.index.unique()
        self.tested_dict = None
        self.tested_df = None
        self.sign_df = None

    def read_file(self, path):
        if path.endswith(".csv"):
            return pd.read_csv(path, sep=";")
        if path.endswith(".xlsx"):
            return pd.read_excel(path)

    def test_all_classes(self, x_col="viability", kstest_for='norm' , save_results=True, return_df=False):
        res_dict = {}
        for sample in self.sample_classes:
            x = self.df_classified.loc[sample][x_col].to_numpy()
            shapiro = self.test_shapiro(x)
            kstest = self.test_ks(x, kstest_for)
            res_dict[sample] = {"shapiro_stat":shapiro[0], "shapiro_pval":shapiro[1],
                                "kstest_stat":kstest[0], "kstest_pval":kstest[1]}
        self.tested_dict = res_dict
        self.tested_df = pd.DataFrame(res_dict).T
        print("Shapiro and KStest was performed for all sample classes")
        print("Results are accessible as dict (var.tested_dict) and dataframe (var.tested_df")
        if save_results:
            self.save_to_csv(self.tested_df, f"shapiro_kstested_{self.name}")
        if return_df:
            return self.tested_df
        return self.tested_dict

    def test_shapiro(self, x):
        from scipy.stats import shapiro
        res = shapiro(x)
        return res

    def test_ks(self, x, cdf):
        from scipy.stats import kstest
        res = kstest(x, cdf)
        return res

    def significant_samples(self, target_attr="shapiro_pval", target_pval=0.05):
        sign = []
        not_sign = []
        for key, values in self.tested_dict.items():
            if values[target_attr] <= target_pval:
                sign.append(key)
            else:
                not_sign.append(key)
        print(f"The following samples are not significiant according to {target_attr}:")
        print(not_sign)
        print("--------------------------------\n\n\n")
        print(f"The following samples are significiant according to {target_attr}:")
        print(sign)
        print("--------------------------------\n\n\n")
        print("returning list of significant samples")
        return sign

    def get_significant_samples(self):
        sign_shapiro = self.significant_samples("shapiro_pval")
        sign_kstest = self.significant_samples("kstest_pval")
        sign = []
        for s in sign_shapiro:
            if s in sign_kstest:
                sign.append(s)
        print("significant in both test are following samples:")
        print(sign)
        self.sign_df = self.df_classified.loc[sign]
        self.tested_df.index = self.tested_df.index.set_names(self.sign_df.index.names)
        self.all_significant = self.sign_df.merge(self.tested_df, how="inner", left_index=True, right_index=True)
        return self.sign_df



    def save_to_csv(self, df, name=None, path="outputs"):
        import os
        if not os.path.exists(path):
            os.mkdir(path)
        if name is None:
            name = self.name
        filename = f"{path}/res_{name}.csv"
        df.to_csv(filename)
        print(f"ShapiroAggregator saved results to {filename} successfully")