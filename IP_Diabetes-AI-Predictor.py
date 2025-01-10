import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
# from bokeh.io import show
import plotly.express as px
from bokeh.embed import components
import os


# Load the dataset
file_path = '/home/americanu/PycharmProjects/pythonProject/diabetes.csv' 
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(file_path, names=column_names)

# Prepare data for training
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_scaled, y)

def generate_pdf(user_data, prediction, filename="diabetes_report.pdf"):
    # Create a canvas for the PDF
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Add a title with a border
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.darkblue)
    c.drawString(1 * inch, height - 1 * inch, "Diabetes Risk Assessment Report")
    c.setFillColor(colors.black)
    c.line(1 * inch, height - 1.1 * inch, 7.5 * inch, height - 1.1 * inch)

    # Add a section header for user details
    c.setFont("Helvetica-Bold", 14)
    y_position = height - 1.5 * inch
    c.drawString(1 * inch, y_position, "User Input Details:")

    # Add user details in a tabular format
    c.setFont("Helvetica", 12)
    y_position -= 0.3 * inch
    for key, value in user_data.items():
        c.drawString(1.2 * inch, y_position, f"{key}: {value}")
        y_position -= 0.2 * inch

    # Add the prediction result with color-coded emphasis
    y_position -= 0.3 * inch
    c.setFont("Helvetica-Bold", 14)
    if prediction == 1:
        result_text = "High Risk of Diabetes"
        c.setFillColor(colors.red)
    else:
        result_text = "Low Risk of Diabetes"
        c.setFillColor(colors.green)
    c.drawString(1 * inch, y_position, f"Prediction Result: {result_text}")
    c.setFillColor(colors.black)

    # Add recommendations in a box
    y_position -= 0.5 * inch
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1 * inch, y_position, "Recommendations:")
    y_position -= 0.3 * inch
    recommendations = [
        "1. Maintain a healthy diet rich in fruits and vegetables.",
        "2. Exercise regularly for at least 30 minutes a day.",
        "3. Monitor blood sugar levels and consult a healthcare professional.",
        "4. Stay hydrated and avoid excessive sugary drinks."
    ]
    c.setFont("Helvetica", 12)
    for recommendation in recommendations:
        c.drawString(1.2 * inch, y_position, recommendation)
        y_position -= 0.2 * inch

    # Add a footer
    y_position -= 0.5 * inch
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(1 * inch, y_position, "This report is generated using AI-based analysis. Consult a healthcare provider for more guidance.")

    # Save the PDF
    c.save()

    # Return the file path
    return os.path.abspath(filename)

# Create Bokeh plot
st.subheader("Interactive Glucose Level Distribution")
source = ColumnDataSource(data=dict(glucose=data['Glucose']))

p = figure(title="Distribution of Glucose Levels", x_axis_label='Glucose Level',
           y_axis_label='Count', tools="pan,zoom_in,zoom_out,reset,save")
p.quad(top=source.data['glucose'], bottom=0, left=source.data['glucose'] - 1,
       right=source.data['glucose'] + 1, fill_color='blue', line_color='white', alpha=0.7)

# Show in Streamlit
script, div = components(p)
st.components.v1.html(f"{script}{div}", height=500)

# Age filter
age_filter = st.slider("Filter by Age Range", min_value=0, max_value=100, value=(20, 50))
filtered_data = data[(data['Age'] >= age_filter[0]) & (data['Age'] <= age_filter[1])]

# Plot filtered glucose data
st.subheader("Filtered Glucose Level Distribution")
fig = px.histogram(
    filtered_data,
    x='Glucose',
    nbins=30,
    title="Filtered Glucose Distribution",
    color_discrete_sequence=['green'],
    opacity=0.7
)
st.plotly_chart(fig)


# Streamlit interface
st.title("Diabetes Risk Prediction with Visual Insights")

st.sidebar.header("User Input Features")
pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=500, value=100)
bmi = st.sidebar.number_input("BMI", min_value=0, max_value=50, value=25)
diabetes_pedigree_function = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.0, value=0.5)
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)

# Button to predict and generate report
if st.button("Predict Diabetes Risk and Generate Report"):
    user_input = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age
    }

    # Prediction
    user_input_list = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
    user_input_scaled = scaler.transform(user_input_list)
    prediction = model.predict(user_input_scaled)

    if prediction == 1:
        st.error("### Result: High Risk of Diabetes")
    else:
        st.success("### Result: Low Risk of Diabetes")

    # Visualize user input as a bar chart
    st.subheader("Your Input Features")
    user_features = pd.DataFrame(
        user_input_list,
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    )
    st.bar_chart(user_features.T)

    # Generate PDF report
    pdf_path = generate_pdf(user_input, prediction[0])
    st.success("Report generated successfully!")

    # Add a download button for the PDF
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="Download Report",
            data=pdf_file,
            file_name="Diabetes_Risk_Report.pdf",
            mime="application/pdf"
        )

# Data Visualization Section
st.header("Data Insights")

# Display the dataset (optional)
if st.checkbox("Show Raw Dataset"):
    st.write(data.head())

# Glucose level distribution
st.subheader("Glucose Level Distribution")
fig, ax = plt.subplots()
sns.histplot(data['Glucose'], kde=True, bins=30, color='blue', ax=ax)
ax.set_title("Distribution of Glucose Levels")
st.pyplot(fig)

# Correlation heatmap
st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Outcome distribution
st.subheader("Outcome Distribution (Diabetic vs. Non-Diabetic)")
outcome_counts = data['Outcome'].value_counts()
outcome_labels = ['Non-Diabetic', 'Diabetic']
fig, ax = plt.subplots()
ax.pie(outcome_counts, labels=outcome_labels, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
ax.set_title("Outcome Distribution")
st.pyplot(fig)

# streamlit run /home/americanu/PycharmProjects/pythonProject/IP_Diabetes-AI-Predictor.py"
