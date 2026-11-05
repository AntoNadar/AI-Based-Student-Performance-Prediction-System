import streamlit as st
import joblib 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load trained model
model = joblib.load("ai_student_model.pkl")
features = joblib.load("student_features.pkl")

# Sidebar
st.sidebar.title("🎓 Student Analytics")
st.sidebar.write("Academic Performance System")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Individual Student Prediction",
        "CSV Upload & Batch Prediction",
        "Visualization & Analytics"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "AI-Based Student Academic Performance Prediction System"
)

# --------------------------------
# PAGE 1: INDIVIDUAL PREDICTION
# --------------------------------

if page == "Individual Student Prediction":

    st.title("Individual Student Prediction")

    st.write(
        "Enter student details to estimate their academic performance."
    )

# 1. Numerical inputs with limits

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=None,
        placeholder="75.00"
    )

    hours_studied = st.number_input(
        "Hours Studied Per Week",
        min_value=0.0,
        max_value=70.0,
        value=None,
        placeholder="15.00"
    )

    sleep_hours = st.number_input(
        "Sleeping Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=None,
        placeholder="8.00"
    )

    previous_scores = st.number_input(
        "Previous Exam Scores",
        min_value=0.0,
        max_value=100.0,
        value=None,
        placeholder="70.00"
    )

    tutoring_sessions = st.number_input(
        "Tutoring Sessions Attended",
        min_value=0,
        max_value=20,
        value=None,
        placeholder="2"
    )

# 2. Categorical inputs using dropdowns

    motivation_level = st.selectbox(
        "Motivation To Study",
        ["Low", "Medium", "High"],
        index=None,
        placeholder="Select Motivation Level"
    )

    parental_involvement = st.selectbox(
        "Parental Involvement in Studies",
        ["Low", "Medium", "High"],
        index=None,
        placeholder="Select Involvement Level"
    )


    access_to_resources = st.selectbox(
        "Access to Study Materials & Resources",
        ["Low", "Medium", "High"],
        index=None,
        placeholder="Select Access Level"
    )

# Convert categories into numerical values

# Prediction button

    if st.button("Predict Exam Score"):

    # Check whether all inputs are provided
        if None in [
            attendance,
            hours_studied,
            sleep_hours,
            previous_scores,
            tutoring_sessions,
            motivation_level,
            parental_involvement,
            access_to_resources
        ]:
            st.warning("Please fill in all student details.")

        else:
        # Convert categories into numerical values
            mapping = {
                "Low": 0,
                "Medium": 0.5,
                "High": 1
            }

            motivation = mapping[motivation_level]
            parental = mapping[parental_involvement]
            resources = mapping[access_to_resources]

        # Make prediction
            prediction = model.predict([[
                attendance, hours_studied, sleep_hours, previous_scores,
                tutoring_sessions, motivation, parental, resources
            ]])

            st.success(f"Estimated Exam Score: {prediction[0]:.2f}")

# --------------------------------
# PAGE 2: CSV UPLOAD & BATCH PREDICTION
# --------------------------------

elif page == "CSV Upload & Batch Prediction":

    st.title("CSV Upload & Batch Prediction")

    st.write(
        "Upload student records to predict their academic performance "
        "and identify students who may benefit from additional support."
    )

    st.info(
        "Upload a CSV file containing the eight required student features. "
        "StudentID and Name are optional."
    )

    # CSV file uploader
    uploaded_file = st.file_uploader(
        "Upload Student CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            data = pd.read_csv(uploaded_file)
            st.session_state["uploaded_student_data"] = data.copy()

            st.subheader("Uploaded Dataset")

            st.write(f"Total Students: {len(data)}")

            st.dataframe(data.head(10), use_container_width=True)

            # Required columns
            required_columns = features

            # Check missing columns
            missing_columns = [
                col for col in required_columns
                if col not in data.columns
            ]

            if missing_columns:

                st.error(
                    f"Missing required columns: {missing_columns}"
                )

            elif data.empty:

                st.error("The uploaded CSV file is empty.")

            else:

                # Work on a separate copy
                input_data = data[required_columns].copy()

                # Identify numeric and categorical columns
                numeric_columns = [
                    'Attendance',
                    'Hours_Studied',
                    'Sleep_Hours',
                    'Previous_Scores',
                    'Tutoring_Sessions'
                ]

                categorical_columns = [
                    'Motivation_Level',
                    'Parental_Involvement',
                    'Access_to_Resources'
                ]

                # Convert numerical columns
                for col in numeric_columns:
                    input_data[col] = pd.to_numeric(
                        input_data[col],
                        errors='coerce'
                    )

                # Check missing and invalid numeric values
                invalid_numeric = input_data[
                    numeric_columns
                ].isnull().any(axis=1)

                # Check numeric ranges
                range_checks = (
                    input_data['Attendance'].between(0, 100) &
                    input_data['Hours_Studied'].between(0, 70) &
                    input_data['Sleep_Hours'].between(0, 24) &
                    input_data['Previous_Scores'].between(0, 100) &
                    input_data['Tutoring_Sessions'].between(0, 20)
                )

                invalid_numeric = invalid_numeric | ~range_checks

                # Encode categorical columns
                mapping = {
                    'Low': 0,
                    'Medium': 0.5,
                    'High': 1
                }

                invalid_categories = pd.Series(
                    False, index=input_data.index
                )

                for col in categorical_columns:

                    # Option A: text labels (Low / Medium / High)
                    from_text = pd.to_numeric(
                        input_data[col]
                        .astype('string')
                        .str.strip()
                        .str.title()
                        .map(mapping),
                        errors='coerce'
                    )

                    # Option B: numeric codes (0 / 0.5 / 1)
                    from_number = pd.to_numeric(
                        input_data[col],
                        errors='coerce'
                    )
                    from_number = from_number.where(
                        from_number.isin([0, 0.5, 1])
                    )

                    # Use whichever format worked; NaN means invalid
                    encoded = from_text.fillna(from_number)

                    invalid_categories |= encoded.isna()

                    input_data[col] = encoded

                # Combine validation checks
                invalid_rows = (
                    invalid_numeric | invalid_categories
                )

                st.subheader("Dataset Validation")

                if invalid_rows.any():

                    st.error(
                        f"{invalid_rows.sum()} student records contain "
                        "missing or invalid values."
                    )

                    st.warning(
                        "Please correct these records in your CSV "
                        "and upload the file again."
                    )

                    st.dataframe(
                        data.loc[invalid_rows],
                        use_container_width=True
                    )

                else:

                    st.success(
                        "All student records passed validation!"
                    )

                    # Arrange features in training order
                    input_data = input_data[features]

                    # Batch prediction
                    if st.button("Predict All Student Scores"):

                        predictions = model.predict(input_data)

                        # Create results dataframe
                        results = data.copy()

                        results['Predicted_Exam_Score'] = np.round(
                            predictions, 2
                        )
                        st.session_state["student_prediction_results"] = results.copy()

                        # Flag students for teacher review
                        review_threshold = 65

                        results['Review_Status'] = np.where(
                            results['Predicted_Exam_Score']
                            < review_threshold,
                            'Needs Teacher Review',
                            'No Immediate Flag'
                        )

                        st.subheader("Prediction Summary")

                        total_students = len(results)

                        students_for_review = (
                            results['Predicted_Exam_Score']
                            < review_threshold
                        ).sum()

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Total Students",
                            total_students
                        )

                        col2.metric(
                            "Average Predicted Score",
                            f"{results['Predicted_Exam_Score'].mean():.2f}"
                        )

                        col3.metric(
                            "Students for Review",
                            int(students_for_review)
                        )

                        # Display all predictions
                        st.subheader("Student Prediction Results")

                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                        # Students requiring attention
                        st.subheader(
                            "Students Who May Benefit From Additional Support"
                        )

                        review_students = results[
                            results['Predicted_Exam_Score']
                            < review_threshold
                        ]

                        if not review_students.empty:

                            st.warning(
                                f"{len(review_students)} students "
                                "have predicted scores below 65."
                            )

                            st.dataframe(
                                review_students,
                                use_container_width=True
                            )

                        else:

                            st.success(
                                "No students were flagged by the "
                                "current review threshold."
                            )

                        st.caption(
                            "A predicted score below 65 is only a "
                            "prompt for teacher review, not a definitive "
                            "judgment of student ability."
                        )

                        # Download prediction results
                        csv_data = results.to_csv(
                            index=False
                        ).encode('utf-8')

                        st.download_button(
                            label="Download Prediction Results",
                            data=csv_data,
                            file_name="student_prediction_results.csv",
                            mime="text/csv"
                        )

        except Exception as e:

            st.error(f"Unable to process CSV file: {e}")

# --------------------------------
# PAGE 3: VISUALIZATION & ANALYTICS
# --------------------------------


elif page == "Visualization & Analytics":

    st.title("Visualization & Analytics")
    st.write("Explore insights from your uploaded student dataset.")

    if "uploaded_student_data" not in st.session_state:
        st.warning("Please upload a CSV file in Page 2 first.")
        st.info("Navigate to CSV Upload & Batch Prediction and upload your dataset.")
        st.stop()

    # ---------- Setup ----------
    REVIEW_THRESHOLD = 65
    LEVELS = ["Low", "Medium", "High"]
    LEVEL_CODES = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
    NUM_FEATURES = [
        "Attendance", "Hours_Studied", "Sleep_Hours",
        "Previous_Scores", "Tutoring_Sessions"
    ]
    CAT_FEATURES = [
        "Motivation_Level", "Parental_Involvement", "Access_to_Resources"
    ]
    LABELS = {
        "Attendance": "Attendance (%)",
        "Hours_Studied": "Hours Studied per Week",
        "Sleep_Hours": "Sleep Hours per Day",
        "Previous_Scores": "Previous Exam Scores",
        "Tutoring_Sessions": "Tutoring Sessions",
        "Motivation_Level": "Motivation",
        "Parental_Involvement": "Parental Involvement",
        "Access_to_Resources": "Access to Resources"
    }

    def to_level(value):
        """Accepts Low/Medium/High text OR 0/0.5/1 codes."""
        if isinstance(value, str):
            value = value.strip().title()
            return value if value in LEVELS else None
        try:
            return {0.0: "Low", 0.5: "Medium", 1.0: "High"}.get(float(value))
        except (TypeError, ValueError):
            return None

    def show(fig):
        st.pyplot(fig)
        plt.close(fig)

    data = st.session_state["uploaded_student_data"].copy()
    results = st.session_state.get("student_prediction_results")

    # Ignore predictions that belong to a previously uploaded file
    if results is not None and len(results) != len(data):
        results = None

    # Use actual scores if present, otherwise predicted scores
    if "Exam_Score" in data.columns:
        df = data.copy()
        score_col, score_label = "Exam_Score", "Actual Exam Score"
    elif results is not None and "Predicted_Exam_Score" in results.columns:
        df = results.copy()
        score_col, score_label = "Predicted_Exam_Score", "Predicted Exam Score"
    else:
        df = data.copy()
        score_col, score_label = None, None

    for col in NUM_FEATURES + ([score_col] if score_col else []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CAT_FEATURES:
        if col in df.columns:
            df[col + "_level"] = pd.Categorical(
                df[col].map(to_level), categories=LEVELS, ordered=True
            )
            df[col + "_code"] = (
                df[col + "_level"].astype(object).map(LEVEL_CODES).astype(float)
            )

    available_num = [c for c in NUM_FEATURES if c in df.columns]
    available_cat = [c for c in CAT_FEATURES if c + "_level" in df.columns]

    tab_overview, tab_drivers, tab_risk, tab_model = st.tabs([
        "Overview", "What Drives Scores", "Students At Risk", "Model Predictions"
    ])

    # ---------- TAB 1: OVERVIEW ----------
    with tab_overview:

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students", len(df))

        if score_col:
            s = df[score_col].dropna()
            c2.metric(f"Average ({score_label.split()[0]})", f"{s.mean():.2f}")
            c3.metric("Highest / Lowest", f"{s.max():.0f} / {s.min():.0f}")
            c4.metric(
                f"Scoring Below {REVIEW_THRESHOLD}",
                f"{(s < REVIEW_THRESHOLD).mean() * 100:.1f}%"
            )

            st.subheader(f"{score_label} Distribution")
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(s, bins=20, edgecolor="black", alpha=0.85)
            ax.axvline(s.mean(), color="green", linestyle="--",
                       label=f"Average ({s.mean():.1f})")
            ax.axvline(REVIEW_THRESHOLD, color="red", linestyle=":",
                       label=f"Review threshold ({REVIEW_THRESHOLD})")
            ax.set_xlabel("Exam Score")
            ax.set_ylabel("Number of Students")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            show(fig)
        else:
            c2.metric("Exam Score", "Not available")
            st.info(
                "Your CSV has no Exam_Score column. Run batch prediction on "
                "Page 2 to unlock the score charts."
            )

        st.subheader("Feature Distribution")
        if available_num:
            dist_feature = st.selectbox(
                "Choose a feature", available_num,
                format_func=lambda c: LABELS[c], key="dist_feature"
            )
            fig, ax = plt.subplots(figsize=(9, 3.5))
            ax.hist(df[dist_feature].dropna(), bins=20, edgecolor="black")
            ax.set_xlabel(LABELS[dist_feature])
            ax.set_ylabel("Number of Students")
            ax.grid(axis="y", alpha=0.3)
            show(fig)

    # ---------- TAB 2: WHAT DRIVES SCORES ----------
    with tab_drivers:

        if not score_col:
            st.info("This tab needs Exam_Score or batch predictions.")
        else:
            enc_cols = available_num + [c + "_code" for c in available_cat]
            enc = df[enc_cols + [score_col]].rename(
                columns=lambda c: c.replace("_code", "")
            )

            # 1. Ranked correlation with score
            st.subheader("Which factors matter most?")
            corr = enc.corr()[score_col].drop(score_col).dropna().sort_values()

            if corr.empty:
                st.info("Not enough data to compute correlations.")
            else:
                fig, ax = plt.subplots(figsize=(9, 4))
                colors = ["#d9534f" if v < 0 else "#2e86de" for v in corr]
                ax.barh([LABELS.get(i, i) for i in corr.index],
                        corr.values, color=colors)
                for i, v in enumerate(corr.values):
                    ax.text(v, i, f" {v:.2f} ", va="center",
                            ha="left" if v >= 0 else "right")
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel(f"Correlation with {score_label}")
                ax.set_xlim(min(-0.1, corr.min() - 0.1), max(corr.max() + 0.1, 0.1))
                ax.grid(axis="x", alpha=0.3)
                show(fig)
                st.caption(
                    "Blue = higher value goes with higher scores; red = the "
                    "opposite. Correlation shows association, not cause."
                )

            st.divider()

            # 2. Feature vs score (average line instead of cluttered dots)
            st.subheader("Score by Feature")
            if available_num:
                feature = st.selectbox(
                    "Choose a feature", available_num,
                    format_func=lambda c: LABELS[c], key="driver_feature"
                )
                sub = df[[feature, score_col]].dropna()

                fig, ax = plt.subplots(figsize=(9, 4.5))
                ax.scatter(sub[feature], sub[score_col], alpha=0.12, s=14,
                           color="gray", label="Students")

                if sub[feature].nunique() > 12:
                    groups = sub.groupby(
                        pd.cut(sub[feature], bins=10), observed=True
                    )[score_col].mean()
                    xs = [iv.mid for iv in groups.index]
                else:
                    groups = sub.groupby(feature)[score_col].mean()
                    xs = list(groups.index)
                ax.plot(xs, groups.values, marker="o", color="#2e86de",
                        linewidth=2, label="Average score")

                if sub[feature].nunique() > 1:
                    slope, intercept = np.polyfit(sub[feature], sub[score_col], 1)
                    xl = np.array([sub[feature].min(), sub[feature].max()])
                    ax.plot(xl, slope * xl + intercept, color="red",
                            linestyle="--", label=f"Trend ({slope:+.2f} per unit)")

                ax.set_xlabel(LABELS[feature])
                ax.set_ylabel(score_label)
                ax.legend()
                ax.grid(alpha=0.3)
                show(fig)

            st.divider()

            # 3. Categorical factors compared on one chart
            st.subheader("Low vs Medium vs High")
            if available_cat:
                fig, ax = plt.subplots(figsize=(9, 4))
                for c in available_cat:
                    g = df.groupby(c + "_level", observed=False)[score_col].mean()
                    ax.plot(LEVELS, g.reindex(LEVELS).values,
                            marker="o", linewidth=2, label=LABELS[c])
                ax.set_ylabel(f"Average {score_label}")
                ax.set_xlabel("Level")
                ax.legend()
                ax.grid(alpha=0.3)
                show(fig)
                st.caption("A steeper line means that factor changes the score more.")

            with st.expander("Full correlation heatmap"):
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(enc.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                            vmin=-1, vmax=1, center=0, linewidths=0.5, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                fig.tight_layout()
                show(fig)

    # ---------- TAB 3: STUDENTS AT RISK ----------
    with tab_risk:

        if not score_col:
            st.info("This tab needs Exam_Score or batch predictions.")
        else:
            st.write(
                f"Share of students scoring below {REVIEW_THRESHOLD} "
                f"({score_label.lower()}), grouped by key habits. "
                "This is a prompt for teacher review, not a judgment of ability."
            )

            def risk_chart(col, bins, labels, container):
                if col not in df.columns:
                    return
                valid = df[score_col].notna() & df[col].notna()
                band = pd.cut(df.loc[valid, col], bins=bins,
                              labels=labels, right=False)
                tmp = pd.DataFrame({
                    "band": band,
                    "risk": df.loc[valid, score_col] < REVIEW_THRESHOLD
                })
                g = tmp.groupby("band", observed=False)["risk"].agg(["mean", "size"])
                with container:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    bars = ax.bar(labels, g["mean"] * 100, color="#d9534f",
                                  edgecolor="black")
                    for b, (pct, n) in zip(bars, zip(g["mean"] * 100, g["size"])):
                        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                                f"{pct:.0f}%\n(n={n})", ha="center", fontsize=9)
                    ax.set_ylim(0, 115)
                    ax.set_xlabel(LABELS[col])
                    ax.set_ylabel("% below threshold")
                    ax.grid(axis="y", alpha=0.3)
                    show(fig)

            left, right = st.columns(2)
            risk_chart("Attendance", [0, 70, 80, 90, 101],
                       ["<70", "70-79", "80-89", "90-100"], left)
            risk_chart("Hours_Studied", [0, 10, 20, 30, 71],
                       ["<10", "10-19", "20-29", "30+"], right)

    # ---------- TAB 4: MODEL PREDICTIONS ----------
    with tab_model:

        if results is None or "Predicted_Exam_Score" not in results.columns:
            st.info(
                "Run 'Predict All Student Scores' on Page 2 "
                "to see prediction analytics."
            )
        else:
            pred = pd.to_numeric(results["Predicted_Exam_Score"], errors="coerce")

            m1, m2, m3 = st.columns(3)
            m1.metric("Students Predicted", int(pred.notna().sum()))
            m2.metric("Average Predicted Score", f"{pred.mean():.2f}")
            m3.metric(f"Flagged (< {REVIEW_THRESHOLD})",
                      int((pred < REVIEW_THRESHOLD).sum()))

            if "Exam_Score" in results.columns:
                from sklearn.metrics import r2_score, mean_absolute_error

                actual = pd.to_numeric(results["Exam_Score"], errors="coerce")
                ok = actual.notna() & pred.notna()

                if ok.sum() >= 2:
                    st.subheader("Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(actual[ok], pred[ok], alpha=0.25, s=14)
                    lo = min(actual[ok].min(), pred[ok].min())
                    hi = max(actual[ok].max(), pred[ok].max())
                    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--",
                            label="Perfect prediction")
                    ax.set_xlabel("Actual Exam Score")
                    ax.set_ylabel("Predicted Exam Score")
                    ax.legend()
                    ax.grid(alpha=0.3)
                    show(fig)

                    a1, a2 = st.columns(2)
                    a1.metric("R² Score", f"{r2_score(actual[ok], pred[ok]):.3f}")
                    a2.metric("Mean Absolute Error",
                              f"{mean_absolute_error(actual[ok], pred[ok]):.2f} points")
                    st.caption(
                        "If this file was also used to train the model, these "
                        "numbers are optimistic."
                    )
            else:
                st.subheader("Predicted Score Distribution")
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.hist(pred.dropna(), bins=20, edgecolor="black")
                ax.axvline(REVIEW_THRESHOLD, color="red", linestyle=":",
                           label=f"Review threshold ({REVIEW_THRESHOLD})")
                ax.set_xlabel("Predicted Exam Score")
                ax.set_ylabel("Number of Students")
                ax.legend()
                ax.grid(axis="y", alpha=0.3)
                show(fig)

    st.caption(
        "Note: All visualizations are generated from the dataset uploaded by "
        "the user. Predictions are estimates and should not be treated as "
        "definitive academic assessments."
    )