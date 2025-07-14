import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io


class DataCleanerDashboard:
    def __init__(self):
        self.df = None
        self.filtered_df = None
        self.load_data()

    def load_data(self):
        """Handle file upload and pasted CSV data"""
        st.title("📊 Data Analytics Dashboard")

        # File upload widget
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        
        # Text area for pasting CSV data
        pasted_csv = st.text_area("Or, paste your CSV data here (comma-separated)", height=250)
        
        if uploaded_file:
            # Handle CSV file upload
            self.df = pd.read_csv(uploaded_file)
            st.subheader("📄 Raw Data")
            st.dataframe(self.df)
            st.write(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
            self.show_data_cleaning_insights()

        elif pasted_csv:
            # Handle pasted CSV data
            try:
                # Convert pasted CSV text to a pandas DataFrame
                self.df = pd.read_csv(io.StringIO(pasted_csv))
                st.subheader("📄 Raw Data (Pasted CSV)")
                st.dataframe(self.df)
                st.write(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
                self.show_data_cleaning_insights()
            except Exception as e:
                st.error(f"Error processing pasted CSV: {e}")

    def show_data_cleaning_insights(self):
        """Show the data cleaning insights like missing values, duplicates, etc."""
        if self.df is not None:
            self.show_missing_values()
            self.show_constant_columns()
            self.show_duplicates()
            self.show_mixed_types()
            self.show_outliers()

    def show_missing_values(self):
        """Highlight missing values in the dataset"""
        st.markdown("#### 🔻 Missing Values")
        missing_values = self.df[self.df.isnull().any(axis=1)]  # Rows with any missing values
        if not missing_values.empty:
            st.warning(f"Found missing values in the following rows:")
            st.dataframe(missing_values)
        else:
            st.success("No missing values found ✅")

    def show_constant_columns(self):
        """Highlight columns with constant values"""
        st.markdown("#### 🟡 Constant Columns")
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() == 1]
        if constant_cols:
            st.warning(f"Columns with constant values: {', '.join(constant_cols)}")
        else:
            st.success("No constant columns found ✅")

    def show_duplicates(self):
        """Highlight duplicate rows in the dataset"""
        st.markdown("#### ⚠️ Duplicate Rows")
        duplicate_rows = self.df[self.df.duplicated()]
        if not duplicate_rows.empty:
            st.warning(f"Found {len(duplicate_rows)} duplicate rows:")
            st.dataframe(duplicate_rows)
        else:
            st.success("No duplicate rows found ✅")

    def show_mixed_types(self):
        """Highlight columns with mixed data types"""
        st.markdown("#### 🧪 Mixed Data Types")
        mixed_cols = []
        for col in self.df.columns:
            try:
                self.df[col].apply(float)
            except:
                try:
                    self.df[col].astype(str).apply(float)
                except:
                    continue
            if self.df[col].map(type).nunique() > 1:
                mixed_cols.append(col)

        if mixed_cols:
            st.warning(f"Columns with mixed data types: {', '.join(mixed_cols)}")
        else:
            st.success("No mixed-type columns found ✅")

    def show_outliers(self):
        """Detect outliers using IQR"""
        st.subheader("🚨 Outlier Detection")

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            col_outlier = st.selectbox("Select Numeric Column for Outlier Detection", numeric_cols)
            if st.button("Detect Outliers"):
                Q1 = self.df[col_outlier].quantile(0.25)
                Q3 = self.df[col_outlier].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = self.df[(self.df[col_outlier] < lower_bound) | (self.df[col_outlier] > upper_bound)]
                st.warning(f"Detected {len(outliers)} outliers in column '{col_outlier}'.")
                st.dataframe(outliers)

    def filter_data(self):
        """Range filter for numeric columns"""
        st.subheader("🔍 Filter Data by Range")

        # Select a numeric column for range filtering
        numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_columns:
            column = st.selectbox("Select Numeric Column to Filter", numeric_columns)

            # Get the min and max values for the selected column
            min_value, max_value = float(self.df[column].min()), float(self.df[column].max())

            # Allow users to select a range of values
            range_filter = st.slider(
                f"Select Range for {column}",
                min_value=min_value,
                max_value=max_value,
                value=(min_value, max_value)
            )

            # Filter data based on the range
            self.filtered_df = self.df[(self.df[column] >= range_filter[0]) & (self.df[column] <= range_filter[1])]
            st.dataframe(self.filtered_df)

    def descriptive_statistics(self):
        """Show descriptive statistics of the data"""
        st.subheader("📊 Descriptive Statistics")
        st.write(self.df.describe())

    def correlation_analysis(self):
        """Display correlation matrix"""
        st.subheader("🔗 Correlation Analysis")
    
        # Select only numeric columns for correlation
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Ensure there are numeric columns to work with
        if len(numeric_cols) > 0:
            corr_matrix = self.df[numeric_cols].corr()
            st.write(corr_matrix)
    
            # Plotting correlation heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for correlation.")
    

    def column_conversion(self):
        """Convert columns between different data types"""
        st.subheader("🔄 Column Conversion")
        column = st.selectbox("Select Column for Conversion", self.df.columns)
        data_type = st.selectbox("Select Data Type to Convert", ["int", "float", "string", "datetime"])

        if st.button(f"Convert {column} to {data_type}"):

            try:
                if data_type == "int":
                    self.df[column] = self.df[column].astype(int)
                elif data_type == "float":
                    self.df[column] = self.df[column].astype(float)
                elif data_type == "string":
                    self.df[column] = self.df[column].astype(str)
                elif data_type == "datetime":
                    self.df[column] = pd.to_datetime(self.df[column])

                st.success(f"Converted {column} to {data_type} type successfully.")
                st.dataframe(self.df)
            except Exception as e:
                st.error(f"Error: {e}")

    def export_to_excel(self, df):
        """Export DataFrame to Excel"""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        return output.getvalue()

    def export_data(self):
        """Export filtered or original data to CSV/Excel"""
        if self.filtered_df is None:
            self.filtered_df = self.df

        # CSV Export
        csv = self.filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='filtered_data.csv',
            mime='text/csv',
        )

        # Excel Export
        excel = self.export_to_excel(self.filtered_df)
        st.download_button(
            label="Download Excel",
            data=excel,
            file_name="filtered_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    def data_cleaning_actions(self):
        """Perform data cleaning actions like dropping missing rows or filling missing values"""
        st.subheader("🧹 Data Cleaning Actions")

        # Option to drop rows with missing values
        if st.button("Drop Rows with Missing Values"):
            self.df = self.df.dropna()
            st.success("Rows with missing values have been dropped.")
            st.dataframe(self.df)

        # Option to fill missing values
        fill_value = st.text_input("Fill Missing Values With (e.g., '0' or 'mean')", "0")
        if st.button(f"Fill Missing Values with {fill_value}"):
            self.df = self.df.fillna(fill_value)
            st.success(f"Missing values have been filled with {fill_value}.")
            st.dataframe(self.df)

        # Option to remove duplicate rows
        if st.button("Remove Duplicate Rows"):
            self.df = self.df.drop_duplicates()
            st.success("Duplicate rows have been removed.")
            st.dataframe(self.df)

    def visualize_data(self):
        """Create data visualizations"""
        st.subheader("📈 Data Visualization")
    
        # Ensure the user has selected valid columns
        if self.df is not None:
            all_columns = self.df.columns.tolist()
    
            # Select chart type
            chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Pie Chart", "Box Plot", "Scatter Plot"])
    
            # Select X and Y columns
            x_col = st.selectbox("X-axis", all_columns)
            y_col = st.selectbox("Y-axis", self.df.select_dtypes(include=['int64', 'float64']).columns.tolist())
    
            # Validate the columns selected
            if x_col not in self.df.columns or y_col not in self.df.columns:
                st.error("Invalid column selected. Please select valid columns for both X and Y axes.")
                return
    
            fig, ax = plt.subplots()
    
            if chart_type == "Bar Chart":
                sns.barplot(x=self.df[x_col], y=self.df[y_col], ax=ax)
            elif chart_type == "Line Chart":
                sns.lineplot(x=self.df[x_col], y=self.df[y_col], ax=ax)
            elif chart_type == "Pie Chart":
                pie_data = self.df[x_col].value_counts()
                ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
                ax.axis("equal")
            elif chart_type == "Box Plot":
                sns.boxplot(x=self.df[x_col], y=self.df[y_col], ax=ax)
            elif chart_type == "Scatter Plot":
                sns.scatterplot(x=self.df[x_col], y=self.df[y_col], ax=ax)
    
            st.pyplot(fig)
    

    def display(self):
        """Display the main content of the dashboard"""
        if self.df is not None:
            self.data_cleaning_actions()
            self.filter_data()
            self.descriptive_statistics()
            self.correlation_analysis()
            self.column_conversion()
            self.export_data()
            self.visualize_data()


# Initialize and run the dashboard
dashboard = DataCleanerDashboard()
dashboard.display()
