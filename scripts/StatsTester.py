import pandas as pd
import scipy.stats


class StatsTestAggregator:
    def __init__(self, path, name="drug_viability"):
        self.name = name
        self.df = self.read_file(path)
        self.df_orig = self.df.copy()
        self.df_classified = self.df.set_index(["expression", "substance", "dose", "timepoint"])
        self.sample_classes = self.df_classified.index.unique()
        self.pgrmc1_classified = self.df_classified.xs("PGRMC1", level="expression")
        self.evc_classified = self.df_classified.xs("EVC", level="expression")
        self.tested_dict = None
        self.tested_df = None
        self.sign_df = None
        self.ttest_results = None
        self.wilkoks = None

    def read_file(self, path):
        if path.endswith(".csv"):
            return pd.read_csv(path, sep=";")
        if path.endswith(".xlsx"):
            return pd.read_excel(path)

    def calculate_all(self, output="outputs"):
        self.shapiro_all_classes(output=output)
        self.ttest_all_classes(output=output)
        self.wilcox_all_classes(output=output)
        self.mannwhitneyu_all_classes(output=output)

    def shapiro_all_classes(self, x_col="viability", kstest_for='norm' , output="outputs"):
        res_dict = {}
        for sample in self.sample_classes:
            x = self.df_classified.loc[sample][x_col].to_numpy()
            shapiro = self.test_shapiro(x)
            kstest = self.test_ks(x, kstest_for)
            res_dict[sample] = {"shapiro_stat":shapiro[0], "shapiro_pval":shapiro[1],
                                "kstest_stat":kstest[0], "kstest_pval":kstest[1]}
        self.tested_dict = res_dict
        self.tested_df = pd.DataFrame(res_dict).T
        self.save_to_csv(self.tested_df, f"shapiro_{self.name}", path=output)
        return self.tested_df

    def ttest_all_classes(self, col="viability", output="outputs"):
        classes = self.pgrmc1_classified.index.unique()
        res_dict = {}
        for cl in classes:
            x = self.pgrmc1_classified.loc[cl][col]
            y = self.evc_classified.loc[cl][col]
            res = self.ttest(x, y)
            res_dict[cl] = {"ttest_stat":res[0], "ttest_pval":res[1]}
        res_df = pd.DataFrame(res_dict).T
        self.save_to_csv(res_df, f"ttest_{self.name}", output)
        return res_df

    def wilcox_all_classes(self, col="viability", output="outputs"):
        classes = self.pgrmc1_classified.index.unique()
        res_dict = {}
        for cl in classes:
            x = self.pgrmc1_classified.loc[cl][col]
            y = self.evc_classified.loc[cl][col]
            res = self.test_wicox(x, y)
            res_dict[cl] = {"wilcox_stat":res[0], "wilcox_pval":res[1]}
        res_df = pd.DataFrame(res_dict).T
        self.save_to_csv(res_df, f"wilcox_{self.name}", output)
        return res_df

    def mannwhitneyu_all_classes(self, col="viability", output="outputs"):
        classes = self.pgrmc1_classified.index.unique()
        res_dict = {}
        for cl in classes:
            x = self.pgrmc1_classified.loc[cl][col]
            y = self.evc_classified.loc[cl][col]
            res = self.test_mannwhitneyu(x, y)
            res_dict[cl] = {"mannwhitney_stat":res[0], "mannwhitney_pval":res[1]}
        res_df = pd.DataFrame(res_dict).T
        self.save_to_csv(res_df, f"mannwhitney_{self.name}", output)
        return res_df


    def ttest(self, x, y):
        from scipy.stats import ttest_rel
        return ttest_rel(x, y)

    def test_shapiro(self, x):
        from scipy.stats import shapiro
        res = shapiro(x)
        return res

    def test_ks(self, x, cdf):
        from scipy.stats import kstest
        res = kstest(x, cdf)
        return res
    def test_wicox(self, x, y):
        from scipy.stats import wilcoxon
        return wilcoxon(x, y)

    def test_mannwhitneyu(self, x, y):
        from scipy.stats import mannwhitneyu
        return mannwhitneyu(x,y)

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
