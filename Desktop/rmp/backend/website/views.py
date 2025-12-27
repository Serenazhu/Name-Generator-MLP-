from flask import render_template, request, redirect, url_for, flash, Blueprint

import pandas as pd

df = pd.read_csv("data/merged.csv")
df_with_course = pd.read_csv("data/merged_class.csv")

views = Blueprint('views', __name__)

@views.route('/')
def home():
    q = request.args.get("q", "")
    
    df["Name"] = df["Name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    names = df["Name"].tolist()
    not_found = request.args.get("not_found", "")

    print(len(names))
    print("query:", q)
    
    return render_template("index.html",names=names, not_found=not_found)

@views.route("/professor")
def professor():
    name = request.args.get("name", "").strip()
    sub = df[df["Name"] == name]
    if sub.empty:
        return redirect(url_for("views.home", not_found=name))
    
    row = sub.iloc[0] 
    total = int(row["Total"]) 
    
    # --- helper: percent of total ---
    def pct(x):
        return (x/total*100)
    
    # --- PIE CHART ---
    A = pct(row.get("A", 0))
    B = pct(row.get("B", 0))
    C = pct(row.get("C", 0))
    D = pct(row.get("D", 0))
    F = pct(row.get("F", 0))
    W = pct(row.get("W", 0))
    IPP = pct(row.get("IPP", 0))
    INP = pct(row.get("INP", 0))
    Other = IPP+INP
    
    THRESH = 1.0  # show only if > 1%

    raw = {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "F": F,
        "W": W,
        "IP": Other,
    }

    chart = {k: round(v, 1) for k, v in raw.items() if v > THRESH}
    
    # --- PROGRESS BARS ---
    A_cnt = int(row["A"])
    B_cnt = int(row["B"])
    C_cnt = int(row["C"])
    D_cnt = int(row["D"])
    F_cnt = int(row["F"])
    W_cnt = int(row["W"])
    DF_cnt = D_cnt + F_cnt 
    bars = [
        ("A", A_cnt),
        ("B", B_cnt),
        ("C", C_cnt),
        ("D&F", DF_cnt),
        ("W", W_cnt),
    ]
    dept = row["Dept"]
    
    # BY COURSE
    sub_course = df_with_course[df_with_course["Name"] == name]
    courses = sorted(sub_course["Course"].tolist())
    
    course_cards = []
    for _, r in sub_course.iterrows():
        c_total = int(r["Total"]) 
        course = r["Course"]
        A = int(r.get("A", 0))
        B = int(r.get("B", 0))
        C = int(r.get("C", 0))
        D = int(r.get("D", 0))
        F = int(r.get("F", 0))
        W = int(r.get("W", 0))
        IPP = int(r.get("IPP", 0))
        INP = int(r.get("INP", 0))
        Other = IPP+INP
        DF = D+F
        def cpct(x):
            return (x/c_total*100)
        c_raw = {
            "A": cpct(A),
            "B": cpct(B),
            "C": cpct(C),
            "D+F": cpct(DF),
            "W": cpct(W),
            "IP": cpct(Other),
        }
        c_chart = {k: round(v, 1) for k, v in c_raw.items() if v > THRESH}
        course_cards.append({
            "course": course,
            "total": c_total,
            "counts": {"A": A, "B": B, "C":C, "D+F": DF, "W": W},
            "chart": c_chart,
        })

    return render_template("professor.html", name=name, dept=dept, chart=chart, bars=bars, total_cnt=total,courses=courses, course_cards=course_cards)
