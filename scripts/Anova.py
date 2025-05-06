from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pingouin as pg
import pandas as pd
import re

def rename(text):
    text = re.sub(r'[^a-zA-Z]', "",  text) 
    return text

# Analysis of Varience (One Way ANOVA)
def One_way_anova(data, Metrics, group_cols):
    results = []
    group_cols = [rename(col) for col in group_cols]
    data = data.rename(columns={col: rename(col) for col in data.columns})
    for group in group_cols:
        for col in Metrics:
            column_name = rename(col)  
            formula = f"{column_name} ~ C({group})" 
            model = smf.ols(formula, data=data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            for source, row in anova_table.iterrows():
                p_value = row["PR(>F)"]
                interpretation = "Significant" if p_value < 0.05 else "No significant"
                if source == "Residual": interpretation = "-"
        
                results.append({
                    "Variable": col,
                    "Factor": group.title(),
                    "Source": source,
                    "Sum Sq": row["sum_sq"],
                    "df": row["df"],
                    "F-Value": row["F"],
                    "p-Value": p_value,
                    "Interpretation": interpretation
                })

    return pd.DataFrame(results)

# Two Way ANOVA (Interaction Effect)
def two_way_anova_all(data, numerical_columns, Factor1, Factor2):
    results = []

    Factor1 = rename(Factor1)
    Factor2 = rename(Factor2)
    data = data.rename(columns={col: rename(col) for col in data.columns})
    
    for response_column in numerical_columns:
        safe_column_name = rename(response_column)
        data = data.rename(columns={response_column: safe_column_name})
        formula = f"{safe_column_name} ~ C({Factor1}) + C({Factor2}) + C({Factor1}):C({Factor2})"
        
        model = ols(formula, data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        for source, row in anova_table.iterrows():
            p_value = row["PR(>F)"]
            interpretation = "Significant difference" if p_value < 0.05 else "No significant difference"
            if source == "Residual":
                interpretation = "-"
                
            results.append({
                "Variable": response_column,
                "Source": source,
                "Sum Sq": row["sum_sq"],
                "df": row["df"],
                "F-Value": row["F"],
                "p-Value": p_value,
                "Interpretation": interpretation
            })

    results_df = pd.DataFrame(results)
    return results_df

# Welch's ANOVA (Welch's F test)
def welchs_anova(data, Metrics, group_cols):
    results = []
    
    group_cols = [rename(col) for col in group_cols]
    data = data.rename(columns={col: rename(col) for col in data.columns})
    for group in group_cols:
        for col in Metrics:
            column_name = rename(col)
            
            # Perform Welch's ANOVA using pingouin
            aov = pg.welch_anova(data=data, dv=column_name, between=group)
            
            for _, row in aov.iterrows():
                p_value = row["p-unc"]
                interpretation = "Significant difference" if p_value < 0.05 else "No significant difference"
                results.append({
                    "Variable": col,
                    "Grouping Factor": group.title(),
                    "Source": row["Source"],
                    "df": row["ddof1"],  # Degrees of freedom between groups
                    "F-Value": row["F"],
                    "p-Value": p_value,
                    "Interpretation": interpretation
                })

    return pd.DataFrame(results)