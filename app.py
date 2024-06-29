import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


def main():
    st.image("img/knn.png", width=100)
    
    with st.sidebar :
        page = option_menu ("Pilih Halaman", ["Home", "Data Understanding","Preprocessing", "Model", "Evaluasi","Testing"], default_index=0)

    if page == "Home":
        show_home()
    elif page == "Data Understanding":
        show_understanding()
    elif page == "Preprocessing":
        show_preprocessing()
    elif page == "Model":
        show_model()
    elif page == "Evaluasi":
        show_evaluasi()
    elif page == "Testing":
        show_testing()

def show_home():
    st.title("Klasifikasi Lokasi TB dengan menggunakan Metode K-Nearest Neighbor")

    # Explain what is Decision Tree
    st.header("Apa itu K-Nearest Neighbor?")
    st.write("K-Nearest Neighbor (KNN) merupakan salah satu algoritma yang digunakan untuk memprediksi kelas atau kategori dari data baru berdasarkan mayoritas kelas dari tetangga terdekat")

    # Explain the purpose of this website
    st.header("Tujuan Website")
    st.write("Website ini bertujuan untuk memberikan pemahaman mengenai tahapan proses pengolahan data dan klasifikasi dengan menggunakan metode KNN.")

    # Explain the data
    st.header("Data")
    st.write("Data yang digunakan adalah Dataset mahasiswa diambil dari website Kaggle.")

    # Explain the process of Decision Tree
    st.header("Tahapan Proses Klasifikasi Perceptron")
    st.write("1. **Data Understanding atau Pemahaman Data**")
    st.write("2. **Preprocessing Data**")
    st.write("3. **Pemodelan**")
    st.write("4. **Evaluasi Model**")
    st.write("5. **Implementasi**")

def show_understanding():
    st.title("Data Understanding")
    data = pd.read_csv("dataset (1).csv")
    
    st.header("Metadata dari dataset Mahasiswa")
    st.dataframe(data)
    
    col1, col2 = st.columns(2,vertical_alignment='top')
    
    with col1 :
        st.write("Jumlah Data : ", len(data.axes[0]))
        st.write("Jumlah Atribut : ", len(data.axes[1]))
    
    with col2 :
        st.write(f"Terdapat {len(data['Target'].unique())} Label Kelas, yaitu : {data['Target'].unique()}")

    st.markdown("---")
    
    st.header("Tipe Data & Missing Value")
    
    r2col1, r2col2 = st.columns(2,vertical_alignment='bottom')
    
    with r2col1 :
        st.write("Tipe Data")
        st.write(data.dtypes)
    
    with r2col2 :
        st.write("Missing Value")
        st.write(data.isnull().sum())
    
    st.markdown("---")
    
    st.header("Eksplorasi Data")
    
    st.dataframe(data.describe())    
    
    st.markdown("---")
    
    target_counts = data['Target'].value_counts()

    st.header('Distribusi Target')

    fig, ax = plt.subplots(figsize=(8, 6))
    target_counts.plot(kind='bar', color='blue', ax=ax)
    ax.set_title('Distribusi Target')
    ax.set_xlabel('Target')
    ax.set_ylabel('Jumlah')
    ax.set_xticks(range(len(target_counts.index)))
    ax.set_xticklabels(target_counts.index, rotation=0)

    st.pyplot(fig)
    
    st.markdown("---")
    
    
    cat_var = ['Previous qualification', 'Mother\'s qualification', 'Father\'s qualification', 'Application order',
           'Marital status', 'Application mode', 'Course', 'Daytime/evening attendance', 'Nacionality',
           'Mother\'s occupation', 'Father\'s occupation', 'Displaced', 'Educational special needs',
           'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']

# Streamlit code to display the count plots
    st.header('Distribusi Fitur Kategorikal berdasarkan Target')

    # Create count plots for each categorical feature
    for cat in cat_var:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=cat, data=data[data["Target"] != "Enrolled"], hue="Target", ax=ax)
        ax.set_title(f'Distribusi {cat} berdasarkan Target')
        ax.legend(loc='upper right')
        st.pyplot(fig)
        plt.close(fig)
    
    st.markdown("---")
    numeric_data = data.select_dtypes(include=['number'])

# Calculate the correlation matrix for all numerical features
    all_features_corr = numeric_data.corr()

    # Define the specific features for the second correlation matrix
    specific_features = ['Course', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
                        'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)',
                        'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']

    # Calculate the correlation matrix for the specific features
    specific_features_corr = numeric_data[specific_features].corr()

    # Streamlit code to display the correlation matrices
    st.header('Correlation Matrices')

    st.subheader('Correlation Matrix untuk semua fitur')
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(all_features_corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix untuk semua fitur')
    st.pyplot(fig)

    st.subheader('Correlation Matrix untuk fitur yang dipilih')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(specific_features_corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix untuk fitur yang dipilih')
    st.pyplot(fig)
    
def show_preprocessing():
    st.title("Preprocessing")
    
    data = pd.read_csv("dataset (1).csv")
    
    all_fitur = ['Marital status', 'Course',
             'Previous qualification',
             "Mother's qualification", "Father's qualification", "Mother's occupation",
             "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor',
             'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment',
             'International', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
             'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
             'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
             'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
             'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
             'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
             'Unemployment rate', 'Inflation rate', 'GDP', 'Target']


    fitur_columns = ['Course', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
                    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
                    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
                    'Target']

    data = data.drop(columns=[col for col in data.columns if col not in fitur_columns])

    st.header("Memilih atribut yang digunakan untuk pemodelan")
    st.dataframe(data)
    st.write("Jumlah Data : ", len(data.axes[0]))
    st.write("Jumlah Atribut : ", len(data.axes[1]))

    st.markdown("---")
    
    st.header("Drop Target yang tidak diperlukan")
    data = data[data['Target'] != 'Enrolled'].reset_index(drop=True)

    st.dataframe(data)
    st.write("Jumlah Data : ", len(data.axes[0]))
    st.write("Jumlah Atribut : ", len(data.axes[1]))
    
    st.markdown("---")
    
    st.header("Normalisasi Data menggunakan Min Max Scalar")
    
    x = data.drop(['Target'], axis=1)
    y = data['Target']

    data = list(x.columns)
    scaler = MinMaxScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    x_scaled = pd.DataFrame(x_scaled, columns = x.columns)
    st.dataframe(x_scaled)
    
    st.session_state['preprocessed_data'] = x_scaled
    st.session_state['target'] = y

def show_model():
    st.title("Testing Model")
    
    if 'preprocessed_data' in st.session_state and 'target' in st.session_state:
        X_scaled = st.session_state['preprocessed_data']
        y = st.session_state['target']
        combined_data = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
        st.dataframe(combined_data)
        
        st.markdown("---")
        
        st.header("Memecah menjadi data Training dan data Testing")
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0, train_size=0.8, shuffle=True)
        
        trained = pd.concat([X_train, y_train], axis=1)
        st.write("### Data Training 80%")
        st.dataframe(trained)
        st.write("Jumlah Data : ", len(trained.axes[0]))
        
        testing = pd.concat([X_test, y_test], axis=1)
        st.write("### Data Testing 20%")
        st.dataframe(testing)
        st.write("Jumlah Data : ", len(testing.axes[0]))
        
        st.markdown("---")
        
        st.header("Testing menggunakan K = 8")
        
        clf_KNN8 = KNeighborsClassifier(n_neighbors=8)
        clf_KNN8.fit(X_train, y_train)

        y_pred_KNN8 = clf_KNN8.predict(X_test)
        df_pred_KNN8 = pd.DataFrame(y_pred_KNN8, columns=["KNN8"])

        df_test = pd.DataFrame(y_test).reset_index(drop=True)

        df_pred_combined = pd.concat([df_pred_KNN8, df_test], axis=1)
        df_pred_combined.columns = ["KNN8", "Actual Class"]

        st.dataframe(df_pred_combined)
        st.session_state['y_pred'] = y_pred_KNN8
        st.session_state['y_test'] = y_test
        
    else :
        st.write("### :red[Buka Menu Preprocessing terlebih dahulu jika halaman tidak menampilkan data]")

def show_evaluasi():
    st.title("Evaluasi Metode KNN")
    
    if 'y_pred' in st.session_state and 'y_test' in st.session_state:
        
        y_pred_KNN8 = st.session_state['y_pred']
        y_test = st.session_state['y_test']
        
        unique_classes = y_test.unique()
        c_matrix = confusion_matrix(y_test, y_pred_KNN8, labels=unique_classes)
        
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=unique_classes).plot(ax=ax)
        plt.title("Confusion Matrix for KNN")
        st.pyplot(fig)
        
        st.markdown("---")
        
        st.write("### Rumus Untuk menentukan Akurasi, Recall, Presisi, dan F1 Score")
        col1, col2 = st.columns(2,vertical_alignment='top')
        
        with col1 :
            st.latex(r'Accuracy = \frac{TP + TN}{TP + TN + FP + FN}')
            
        
        with col2 :
            st.latex(r'Precision = \frac{TP}{TP + FP}')
        
        col1, col2 = st.columns(2,vertical_alignment='top')
        
        with col1 :
            st.latex(r'F1 Score = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}')
            
        with col2 :
            st.latex(r'Recall = \frac{TP}{TP + FN}')
            
        st.markdown("---")
        
        accuracy = accuracy_score(y_test, y_pred_KNN8) * 100
        precision = precision_score(y_test, y_pred_KNN8, average='weighted') * 100
        recall = recall_score(y_test, y_pred_KNN8, average='weighted') * 100
        f1 = f1_score(y_test, y_pred_KNN8, average='weighted') * 100
        
        
        st.write("### Performance Metrics for KNN Model")
        col1, col2 = st.columns(2,vertical_alignment='top')

        with col1 :
            st.write(f"#### Accuracy: {accuracy:.2f}%")
            
        with col2 :
            st.write(f"#### Precision: {precision:.2f}%")
        
        col1, col2 = st.columns(2,vertical_alignment='top')
        
        with col1 :
            st.write(f"#### Recall: {recall:.2f}%")
        
        with col2 :
            st.write(f"#### F1 Score: {f1:.2f}%")
        
        st.markdown("---")
        
        report = classification_report(y_test, y_pred_KNN8)
        
        st.markdown(f"```\n{report}\n```")
    else :
        st.write("### :red[Buka Menu Preprocessing terlebih dahulu jika halaman tidak menampilkan data]")

def show_testing():
    st.title("Testing Model")
    st.title("Student Graduation Prediction")

    with open("model/knn_pickle", "rb") as r:
        knnp = pickle.load(r)

    with open("model/scaler_pickle", "rb") as s:
        scaler = pickle.load(s)

    LABEL = ["Dropout", "Graduate"]
    
    course_options = [
        "Biofuel Production Technologies",
        "Animation and Multimedia Design",
        "Social Service (evening attendance)",
        "Agronomy",
        "Communication Design",
        "Veterinary Nursing",
        "Informatics Engineering",
        "Equiniculture",
        "Management",
        "Social Service",
        "Tourism",
        "Nursing",
        "Oral Hygiene",
        "Advertising and Marketing Management",
        "Journalism and Communication",
        "Basic Education",
        "Management (evening attendance)"
    ]
    
    course = st.selectbox('Course', options=list(range(1, len(course_options) + 1)), format_func=lambda x: course_options[x - 1])
    
    bayar = st.selectbox('Tuition Fees', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    gender = st.selectbox('Gender', [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    
    bea = st.selectbox('Scholarship Holder', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    ip1 = st.number_input('IP1', min_value=0.0, step=0.01)
    nilai1 = st.number_input('Nilai1', min_value=0.0, step=0.01)
    ip2 = st.number_input('IP2', min_value=0.0, step=0.01)
    nilai2 = st.number_input('Nilai2', min_value=0.0, step=0.01)

    if st.button('Predict'):
        newdata = [[course, bayar, gender, bea, ip1, nilai1, ip2, nilai2]]
        newdata_scaled = scaler.transform(newdata)
        result = knnp.predict(newdata_scaled)
        result = LABEL[result[0]]

        col1, col2 = st.columns(2,vertical_alignment='top')
        
        with col1 :
            st.write(f"Course: {course_options[course - 1]}")
            st.write(f"Tuition Fees: {'Yes' if bayar == 1 else 'No'}")
            st.write(f"Gender: {'Male' if gender == 1 else 'Female'}")
            st.write(f"Scholarship Holder: {'Yes' if bea == 1 else 'No'}")
            st.write(f"IP1: {ip1}")
            st.write(f"Nilai1: {nilai1}")
            st.write(f"IP2: {ip2}")
            st.write(f"Nilai2: {nilai2}")
        
        with col2 :
            st.write(f"Prediction: :blue[{result}]")
    

if __name__ == "__main__":
    st.set_page_config(page_title="K-Nearest Neighbor", page_icon="img/knn.png")
    main()
