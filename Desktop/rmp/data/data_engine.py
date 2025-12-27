import pandas as pd
from pathlib import Path


def clean(file_path):
    
    raw = pd.read_excel(file_path, sheet_name=0, header=None, dtype=str)
    raw = raw.iloc[4:].reset_index(drop=True)
    raw = raw.iloc[:-2]

    
    if raw.shape[1] == 16:
        raw.columns = [
        "Division Name","Dept","Course","Name","Meth","Succ.","Compl",
        "A","B","C","IPP","D","F","INP","W","Total"
        ]
    else: 
        raw.columns = [
        "Division Name","Dept_1", "Dept_2","Course","Name","Meth","Succ.","Compl",
        "A","B","C","IPP","D","F","INP","W","Total"
        ]
        raw["Dept"] = raw["Dept_1"].fillna(raw["Dept_2"])
        raw = raw.drop(columns=["Dept_1", "Dept_2"])
    
    cols = raw.columns.tolist()
    cols.remove("Dept")
    cols.insert(1, "Dept")
    raw = raw[cols]

    raw = raw[raw["Course"] != "Course"] # heading on each page

    raw = raw.ffill()
    raw = raw.replace("-", 0)
    
    #raw = raw.drop(columns=["INP", "IPP"])
    
    sum_cols = ["A","B","C","D","F","W","INP", "IPP","Total"]
    raw[sum_cols] = raw[sum_cols].apply(pd.to_numeric)
    
    raw['Succ.'] = pd.to_numeric(raw['Succ.'], errors="coerce")
    raw['Compl'] = pd.to_numeric(raw['Compl'], errors="coerce")
    raw = raw.dropna(subset=["Succ.", "Compl"])
        
    raw = raw[~raw["Division Name"].str.contains("total", case=False, na=False)] # summary for each department
    
    raw["Name"] = raw["Name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    raw["Course"] = raw["Course"].astype(str).str.replace(r"\s+", "", regex=True).str.strip()

    raw = raw.drop_duplicates()

    cols = ["A", "B", "C"]
    drop = ((raw[cols] % 1) != 0).any(axis=1)
    raw = raw.loc[~drop].copy()
    
    raw = raw[raw["Total"] != 0].copy()
    raw = raw[~raw["Course"].str.endswith("S")]

    return raw

def merge(all_data, group_class=False):
    if group_class == False:
        groupby = "Name"
    else:
        groupby = ["Name", "Course"]
    all_df = pd.concat(all_data, ignore_index=True)
    sum_cols = ["A","B","C","D","F","W", "IPP", "INP", "Total"]
    rate_cols = ["Compl","Succ."]
    
    merged = (
        all_df.groupby(groupby, as_index=False)
        .agg({
            "Dept": "first", 
            **{c: "sum" for c in sum_cols},
            **{c: "mean" for c in rate_cols},}
        )
    )
    return merged
    
folder = Path("data/excel")

all_data = []
for path in folder.iterdir():
    df = clean(path)
    filename = path.name
    df.to_csv(f"data/csv/{filename}.csv", index=False)
    all_data.append(df)
merged_ds = merge(all_data,group_class=False)
merged_ds.to_csv("data/merged.csv", index=False)

merged_class_ds = merge(all_data,group_class=True)
merged_class_ds.to_csv("data/merged_class.csv", index=False)
  

# df1 = clean("data/excel/Summer_2024.xlsx")
# df2 = clean("data/excel/Fall_2024.xlsx")
# df3 = clean("data/excel/Spring_2025.xlsx")
# df4 = clean("data/excel/Winter_2025.xlsx")

# df5 = clean("data/excel/SP_2024.xlsx")
# df6 = clean("data/excel/WI_2024.xlsx")
# df7 = clean("data/excel/FA_2023.xlsx")
# df8 = clean("data/excel/SU_2023.xlsx")

# df1.to_csv("data/csv/SU_2024.csv", index=False)
# df2.to_csv("data/csv/FA_2024.csv", index=False)
# df3.to_csv("data/csv/SP_2025.csv", index=False)
# df4.to_csv("data/csv/WI_2025.csv", index=False)

# df5.to_csv("data/csv/SP_2024.csv", index=False)
# df6.to_csv("data/csv/WI_2024.csv", index=False)
# df7.to_csv("data/csv/FA_2023.csv", index=False)
# df8.to_csv("data/csv/SU_2023.csv", index=False)

# all_data = [df1,df2,df3,df4,df5,df6,df7,df8]
# merged_ds = merge(all_data)
# merged_ds.to_csv("data/merged.csv", index=False)
# df1.to_csv("FA_2024.csv", index=False)
# prof_rows = df[df["Name"] == "Murdock, Adam"]
# print(prof_rows)