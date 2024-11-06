import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import Flask, request, send_file, render_template
from prettytable import PrettyTable
import tempfile

app = Flask(__name__)

# Function to create a variety of EDA charts
def create_charts(dataframe, top_n=10):
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns
    chart_list = []

    # Univariate Analysis: Histograms and Box Plots for numerical columns
    for column in numerical_columns:
        plt.figure(figsize=(10, 8))
        sns.histplot(dataframe[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}', color='blue')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            plt.savefig(tmpfile.name)
            chart_list.append(tmpfile.name)
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.boxplot(y=dataframe[column])
        plt.title(f'Box Plot of {column}', color='blue')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            plt.savefig(tmpfile.name)
            chart_list.append(tmpfile.name)
        plt.close()

    # Categorical Analysis: Bar Charts and Pie Charts
    for column in categorical_columns:
        counts = dataframe[column].value_counts()
        top_counts = counts.nlargest(top_n)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_counts.index, y=top_counts.values)
        plt.title(f'Top {top_n} Categories in {column}', color='blue')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            plt.savefig(tmpfile.name)
            chart_list.append(tmpfile.name)
        plt.close()

        if len(counts) <= 20:
            plt.figure(figsize=(10, 8))
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
            plt.title(f'Pie Chart of {column}', color='blue')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                plt.savefig(tmpfile.name)
                chart_list.append(tmpfile.name)
            plt.close()

    # Bivariate Analysis: Scatter Plots
    for i, col1 in enumerate(numerical_columns):
        for col2 in numerical_columns[i + 1:]:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=dataframe[col1], y=dataframe[col2])
            plt.title(f'Scatter Plot between {col1} and {col2}', color='blue')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                plt.savefig(tmpfile.name)
                chart_list.append(tmpfile.name)
            plt.close()

    # Correlation Heatmap
    if len(numerical_columns) > 1:
        plt.figure(figsize=(12, 10))
        sns.heatmap(dataframe.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap', color='blue')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            plt.savefig(tmpfile.name)
            chart_list.append(tmpfile.name)
        plt.close()

    return chart_list

# Function to generate insights summary from the dataset
def generate_insights(dataframe):
    insights = []

    # Summary of numerical data
    if not dataframe.empty:
        insights.append(f"**Dataset Overview**:\n- Rows: {dataframe.shape[0]}, Columns: {dataframe.shape[1]}\n")

        # General statistics
        stats = dataframe.describe(include='all')
        insights.append("**Summary Statistics**:\n")

        summary_table = PrettyTable()
        summary_table.field_names = ["Column", "Statistic", "Value"]

        # Adding summary statistics to the table
        for col in stats.columns:
            for index, value in stats[col].iteritems():
                summary_table.add_row([col, index, value])

        insights.append(str(summary_table) + "\n")

        # Check correlations
        if len(dataframe.select_dtypes(include=['float64', 'int64']).columns) > 1:
            correlation_matrix = dataframe.corr()
            high_correlation = correlation_matrix[correlation_matrix.abs() > 0.5]
            insights.append("**High Correlations (|r| > 0.5)**:\n")

            # Create a PrettyTable for high correlations
            high_corr_table = PrettyTable()
            high_corr_table.field_names = ["Variable 1", "Variable 2", "Correlation Coefficient"]

            for col1 in high_correlation.columns:
                for col2 in high_correlation.index:
                    if col1 != col2 and not pd.isnull(high_correlation.loc[col1, col2]):
                        high_corr_table.add_row([col1, col2, round(high_correlation.loc[col1, col2], 2)])

            insights.append(str(high_corr_table) + "\n")
        else:
            insights.append("Insufficient numerical data to calculate correlations.\n")

        # Categorical insights
        insights.append("**Top Categorical Insights**:\n")
        for column in dataframe.select_dtypes(include=['object', 'category']).columns:
            counts = dataframe[column].value_counts()
            insights.append(f"- Top values for '{column}':\n{counts.head(5).to_string()}\n")
    else:
        insights.append("The dataset is empty.\n")

    return "\n".join(insights)

# Function to create the PDF report with charts and insights
def create_pdf_with_charts_and_insights(pdf_filename, chart_list, insights):
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter
    margin = 40

    # Add charts to the PDF
    for chart_path in chart_list:
        c.drawImage(chart_path, margin, margin, width=width - 2 * margin, height=height - 2 * margin)
        c.showPage()  # Move to a new page for the next chart

    # Add insights summary on the last page
    c.setFont("Helvetica-Bold", 14)
    c.setFillColorRGB(0, 0, 1)  # Blue color for headings
    c.drawString(margin, height - margin, "Summary of Insights")
    c.setFont("Helvetica", 10)  # Smaller font size
    c.setFillColorRGB(0, 0, 0)  # Black color for text
    y_position = height - margin - 30

    for line in insights.split('\n'):
        c.drawString(margin, y_position, line)
        y_position -= 15
        if y_position < 40:  # Create a new page if the space is insufficient
            c.showPage()
            y_position = height - margin - 30

    c.save()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Read the uploaded file into a DataFrame
    try:
        if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            dataframe = pd.read_excel(file)
        elif file.filename.endswith('.csv'):
            dataframe = pd.read_csv(file)  # Ensure CSV files are correctly handled
        else:
            return "Unsupported file format. Please upload a .xls, .xlsx, or .csv file."
    except Exception as e:
        return f"Error reading file: {str(e)}"
    
    # Generate charts and insights
    chart_list = create_charts(dataframe, top_n=10)
    insights = generate_insights(dataframe)

    # Create a PDF report
    pdf_filename = "eda_report.pdf"
    create_pdf_with_charts_and_insights(pdf_filename, chart_list, insights)

    # Send the PDF to the user
    return send_file(pdf_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
