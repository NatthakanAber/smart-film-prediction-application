import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from pickle import dump, load
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder


# Define a function that gets colour information from images¶
def get_colour_info(im):
    #im = Image.open(image, 'r')
    #width, height = im.size
    pixel_values = list(im.getdata())

    # extract the first 3 channels from each pixel
    rgb = [i[:3] for i in pixel_values]
    # exclude white pixels
    rgb = [i for i in rgb if i != (255, 255, 255)]
    # exclude black pixels
    rgb = [i for i in rgb if i != (0, 0, 0)]

    rgb = np.array(rgb)
    q1_r, q1_g, q1_b = np.percentile(rgb, 25, axis=0)  # Q1
    q2_r, q2_g, q2_b = np.percentile(rgb, 50, axis=0)  # median
    q3_r, q3_g, q3_b = np.percentile(rgb, 75, axis=0)  # Q3

    return q1_r, q1_g, q1_b, q2_r, q2_g, q2_b, q3_r, q3_g, q3_b

#---- MAIN ------
st.set_page_config(
    page_title='Smart Film App',
    page_icon = '🐮',
    layout = 'centered',
    menu_items = {
        "About": 'This application is a part of the "xxx." project.'
    }
)

st.title("Smart Film Prediction")

# load best model
#model_name = 'EfficientNet_B0'
#model_path = 'models/'+ model_name + '.pth' #os.path.join('models/', model_name + '.pth')
#model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))

source = {
    'Blueberry': 'BB',
    'Butterfly pea flower': 'B',
    'Cabbage': 'C',
    'Hyacinth bean': 'H',
    'Mulberry': 'M',
    'Roselle': 'R',
}

matrix = {
    'Chitosan': 'C',
    'Chitosan and Gelatin': 'CG',
    'Guar gum and polyvinyl alcohol': 'GP',
    'Polyvinyl alcohol/Cellulose nanocrystals': 'PVA/CNC',
    'Sodium alginate': 'S',
    'Sodium alginate/Gellan gum': 'SA/GG',
    'Wheat gluten protein/apple pectin': 'WG/AP',
}

contact = {
    'Ammonia': 'A',
    'pH': 'pH'
}

# Load the class encoder (lable encoding model trained on the dataset in the pre-processing step)
label_enc = load(open('models/class_encoder.pkl', 'rb'))
class_names = list(label_enc.classes_)
#st.write(class_names)
# load model (SVC)
model = load(open('models/classification_model.pkl', 'rb')) # SVC

with st.form(key='input_form'):
    uploaded_image = st.file_uploader(label="Upload an image (file types: 'tif', 'jpg', 'jpeg', 'HEIC', 'png'):", key='uploaded_image', type=['tif', 'jpg', 'jpeg', 'HEIC', 'png'], accept_multiple_files=False)

    #if uploaded_image is not None:
    #    # show image
    #    img_container = st.container(border=True)  # ,horizontal=True, horizontal_alignment="right")
    #    img_container.image(image=uploaded_image)
    #    image = Image.open((uploaded_image))

    # Concentration
    concentration = st.number_input("Concentration:")
    #st.write("The current number is ", concentration)

    source_option = st.selectbox(
        "Source of Anthocyanin:",
        (source.keys()),
    )
    #st.write("You selected:", source[source_option])
    abb_of_source = source[source_option]

    matrix_option = st.selectbox(
        "Matrix:",
        (matrix.keys()),
    )
    #st.write("You selected:", matrix[matrix_option])
    abb_of_matrix = matrix[matrix_option]

    contact_option = st.selectbox(
        "Contact with:",
        (contact.keys()),
    )
    #st.write("You selected:", contact[contact_option])
    abb_of_contact = contact[contact_option]

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    #st.write('Submitted')
    if uploaded_image is not None:

        # begin prediction process
        image = Image.open((uploaded_image))
        # Image processing >> transform to tabular data
        q1_r, q1_g, q1_b, q2_r, q2_g, q2_b, q3_r, q3_g, q3_b = get_colour_info(image)
        new_data = {
            'q1_r': q1_r,
            'q1_g': q1_g,
            'q1_b': q1_b,
            'q2_r': q2_r,
            'q2_g': q2_g,
            'q2_b': q2_b,
            'q3_r': q3_r,
            'q3_g': q3_g,
            'q3_b': q3_b,
            'Concentration': concentration,
            'Abbreviation of source': abb_of_source,
            'Abbreviation of matrix': abb_of_matrix,
            'Abbreviation of contact': abb_of_contact,
        }
        new_X = pd.DataFrame(new_data, index =[0])
        #st.dataframe(new_X)

        # --------
        # One hot encoding for 'Abbreviation of source', 'Abbreviation of matrix', 'Abbreviation of contact'
        # Transform nominal to numerical features
        # Load the scaler (normalisation model trained on the dataset in the pre-processing step)
        enc = load(open('models/onehot_encoder.pkl', 'rb'))

        # Define a list of nominal features
        nomial_features = ['Abbreviation of source', 'Abbreviation of matrix',
                           'Abbreviation of contact']  # If there are more nominal features, add their column names to the list.

        # Applying OneHotEncoder on the defined nominal features
        encoding_array = enc.transform(new_X[nomial_features])

        # The encoder returns the transformed data as an array, so we need to convert it back to a DataFrame
        encoding_df = pd.DataFrame(encoding_array, columns=enc.get_feature_names_out(nomial_features))

        # Finally, the transformed features are concatenated to the original dataset
        new_X = pd.concat([new_X, encoding_df], axis=1)
        # Define X with all numeric features
        new_X = new_X[['q1_r', 'q1_g', 'q1_b', 'q2_r', 'q2_g', 'q2_b', 'q3_r', 'q3_g', 'q3_b',
                   'Concentration', 'Abbreviation of source_B',
                   'Abbreviation of source_BB', 'Abbreviation of source_C',
                   'Abbreviation of source_H', 'Abbreviation of source_M',
                   'Abbreviation of source_R', 'Abbreviation of matrix_C',
                   'Abbreviation of matrix_CG', 'Abbreviation of matrix_GP',
                   'Abbreviation of matrix_PVA/CNC', 'Abbreviation of matrix_S',
                   'Abbreviation of matrix_SA/GG', 'Abbreviation of matrix_WG/AP',
                   'Abbreviation of contact_A', 'Abbreviation of contact_pH']]
        #st.dataframe(new_X)

        # --------
        # Perform normalisation
        # Normalisation
        # Load the scaler (normalisation model trained on the dataset in the pre-processing step)
        scaler = load(open('models/scaler.pkl', 'rb'))

        # Normalise the X_new dataframe using the normalisation model
        X_scaled = scaler.transform(new_X)
        new_X = pd.DataFrame(X_scaled, columns=new_X.columns)
        #st.dataframe(new_X)


        # --------
        # predict the class of a new image
        # add progress bar !!!
        with st.spinner("Prediction progress, please wait ..."):#, show_time=True):
            # Use the model to predict class labels of the new data sample
            predicted_class = model.predict(new_X)

            st.subheader('This smart film indicates that the meat is "' + class_names[int(predicted_class)] +'".')
            col1, col2 = st.columns(2)
            with col1:

                img_container = st.container(border=True)  # ,horizontal=True, horizontal_alignment="right")
                img_container.image(image=uploaded_image)

            with col2:
                data = {
                    'Concentration': concentration,
                    'Source': source_option,
                    'Matrix': matrix_option,
                    'Contact': contact_option,
                }
                st.table(data)

