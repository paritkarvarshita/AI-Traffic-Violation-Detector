from flask import Flask, render_template, send_from_directory
import os
import pandas as pd

app = Flask(__name__)

VIOLATION_FOLDER = "violations"
REPORT_FILE = "reports/violations.csv"


# =========================
# serve violation images
# =========================
@app.route('/violations/<path:filename>')
def violation_files(filename):
    return send_from_directory(VIOLATION_FOLDER, filename)

@app.route('/reports/<path:filename>')
def report_files(filename):
    return send_from_directory('reports', filename)

# =========================
# dashboard
# =========================
@app.route("/")
def index():

    images = []
    total = 0
    helmet = 0
    signal = 0
    fine = 0

    # load images
    if os.path.exists(VIOLATION_FOLDER):
        files = os.listdir(VIOLATION_FOLDER)
        images = ["violations/" + f for f in files]

    # load csv
    if os.path.exists(REPORT_FILE):
        df = pd.read_csv(REPORT_FILE)

        total = len(df)

        # auto detect violation column safely
        col = df.columns[1]

        helmet = len(df[df[col] == "No Helmet"])
        signal = len(df[df[col] == "Signal Jump"])

        fine = helmet * 500 + signal * 1000

    return render_template(
        "index.html",
        images=images,
        total=total,
        helmet=helmet,
        signal=signal,
        fine=fine
    )


# =========================
# run
# =========================
if __name__ == "__main__":
    app.run(debug=True)
