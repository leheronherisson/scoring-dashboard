# ------------------------------------
# import packages
# ------------------------------------
import requests
import json
from pandas import json_normalize
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import shap
from shap.plots import waterfall
import matplotlib.pyplot as plt
from PIL import Image
# ----------------------------------------------------
# main function
# ----------------------------------------------------
def main():
    # ------------------------------------------------
    # local API (à remplacer par l'adresse de l'application déployée)
    # -----------------------------------------------
    API_URL = "https://backheron.herokuapp.com/app/"
    # Local URL: http://localhost:8501
    # -----------------------------------------------
    # Configuration of the streamlit page
    # -----------------------------------------------
    st.set_page_config(page_title='Tableau de bord de notation des demandes de prêt bancaire',
                       page_icon='🧊',
                       layout='centered',
                       initial_sidebar_state='auto')
    # Display the title
    st.title('prêt bancaire - application - scoring dashboard')
    st.subheader("RCS -OC - Data Scientist")

    # Display the LOGO
    # files = os.listdir('Image_logo')
    # for file in files:
    img = Image.open("LOGO.png")
    st.sidebar.image(img, width=250)

    # # Display the loan image
    # files = os.listdir('Image_loan')
    # for file in files:
    img = Image.open("loan.png")
    st.image(img, width=100)

    # Functions
    # ----------
    def get_list_display_features(f, def_n, key):
        all_feat = f
        n = st.slider("Nb of features to display",
                      min_value=2, max_value=20,
                      value=def_n, step=None, format=None, key=key)

        disp_cols = list(get_features_importances().sort_values(ascending=False).iloc[:n].index)

        box_cols = st.multiselect(
            'Choose the features to display:',
            sorted(all_feat),
            default=disp_cols, key=key)
        return box_cols

    ###############################################################################
    #                      LIST OF API REQUEST FUNCTIONS
    ###############################################################################
    # Get list of ID (cached)
    @st.cache(suppress_st_warning=True)
    def get_id_list():
        # URL of the sk_id API
        id_api_url = API_URL + "id/"
        # Requesting the API and saving the response
        response = requests.get(id_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        id_customers = pd.Series(content['data']).values
        return id_customers

    # Get selected customer's data (cached)
    # local test api : http://127.0.0.1:5000/app/data_cust/?SK_ID_CURR=165690
    data_type = []

    @st.cache
    def get_selected_cust_data(selected_id):
        # URL of the sk_id API
        data_api_url = API_URL + "data_cust/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(data_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        x_custom = pd.DataFrame(content['data'])
        # x_cust = json_normalize(content['data'])
        y_customer = (pd.Series(content['y_cust']).rename('TARGET'))
        # y_customer = json_normalize(content['y_cust'].rename('TARGET'))
        return x_custom, y_customer

    @st.cache
    def get_all_cust_data():
        # URL of the sk_id API
        data_api_url = API_URL + "all_proc_data_tr/"
        # Requesting the API and saving the response
        response = requests.get(data_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))  #
        x_all_cust = json_normalize(content['X_train'])  # Results contain the required data
        y_all_cust = json_normalize(content['y_train'].rename('TARGET'))  # Results contain the required data
        return x_all_cust, y_all_cust

    # Get score (cached)
    @st.cache
    def get_score_model(selected_id):
        # URL of the sk_id API
        score_api_url = API_URL + "scoring_cust/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(score_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # Getting the values of "ID" from the content
        score_model = (content['score'])
        threshold = content['thresh']
        return score_model, threshold

    # Get list of shap_values (cached)
    # local test api : http://127.0.0.1:5000/app/shap_val//?SK_ID_CURR=10002
    @st.cache
    def values_shap(selected_id):
        # URL of the sk_id API
        shap_values_api_url = API_URL + "shap_val/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(shap_values_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        shapvals = pd.DataFrame(content['shap_val_cust'].values())
        expec_vals = pd.DataFrame(content['expected_vals'].values())
        return shapvals, expec_vals

    #############################################
    #############################################
    # Get list of expected values (cached)
    @st.cache
    def values_expect():
        # URL of the sk_id API
        expected_values_api_url = API_URL + "exp_val/"
        # Requesting the API and saving the response
        response = requests.get(expected_values_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        expect_vals = pd.Series(content['data']).values
        return expect_vals

    # Get list of feature names
    @st.cache
    def feat():
        # URL of the sk_id API
        feat_api_url = API_URL + "feat/"
        # Requesting the API and saving the response
        response = requests.get(feat_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        features_name = pd.Series(content['data']).values
        return features_name

    # Get the list of feature importances (according to lgbm classification model)
    @st.cache
    def get_features_importances():
        # URL of the aggregations API
        feat_imp_api_url = API_URL + "feat_imp/"
        # Requesting the API and save the response
        response = requests.get(feat_imp_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert back to pd.Series
        feat_imp = pd.Series(content['data']).sort_values(ascending=False)
        return feat_imp

    # Get data from 20 nearest neighbors in train set (cached)
    @st.cache
    def get_data_neigh(selected_id):
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        neight_data_api_url = API_URL + "neigh_cust/?SK_ID_CURR=" + str(selected_id)
        # save the response of API request
        response = requests.get(neight_data_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame and pd.Series
        # targ_all_cust = (pd.Series(content['target_all_cust']).rename('TARGET'))
        # target_select_cust = (pd.Series(content['target_selected_cust']).rename('TARGET'))
        # data_all_customers = pd.DataFrame(content['data_all_cust'])
        data_neig = pd.DataFrame(content['data_neigh'])
        target_neig = (pd.Series(content['y_neigh']).rename('TARGET'))
        return data_neig, target_neig

    # Get data from 1000 nearest neighbors in train set (cached)
    @st.cache
    def get_data_thousand_neigh(selected_id):
        thousand_neight_data_api_url = API_URL + "thousand_neigh/?SK_ID_CURR=" + str(selected_id)
        # save the response of API request
        response = requests.get(thousand_neight_data_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame and pd.Series
        # targ_all_cust = (pd.Series(content['target_all_cust']).rename('TARGET'))
        # target_select_cust = (pd.Series(content['target_selected_cust']).rename('TARGET'))
        # data_all_customers = pd.DataFrame(content['data_all_cust'])
        data_thousand_neig = pd.DataFrame(content['X_thousand_neigh'])
        x_custo = pd.DataFrame(content['x_custom'])
        target_thousand_neig = (pd.Series(content['y_thousand_neigh']).rename('TARGET'))
        return data_thousand_neig, target_thousand_neig, x_custo

    #############################################################################
    #                          Selected id
    #############################################################################
    # list of customer's ID's
    cust_id = get_id_list()
    # Selected customer's ID
    selected_id = st.sidebar.selectbox('Select customer ID from list:', cust_id, key=18)
    st.write('Your selected ID = ', selected_id)

    ############################################################################
    #                           Graphics Functions
    ############################################################################
    # Global SHAP SUMMARY
    @st.cache
    def shap_summary():
        return shap.summary_plot(shap_vals, feature_names=features)

    # Local SHAP Graphs
    @st.cache
    def waterfall_plot(nb, ft, expected_val, shap_val):
        return shap.plots._waterfall.waterfall_legacy(expected_val, shap_val[0, :],
                                                      max_display=nb, feature_names=ft)

    # Local SHAP Graphs
    @st.cache(allow_output_mutation=True)  #
    def force_plot():
        shap.initjs()
        return shap.force_plot(expected_vals[0][0], shap_vals[0, :], matplotlib=True)

    # Gauge Chart
    @st.cache
    def gauge_plot(scor, th):
        scor = int(scor * 100)
        th = int(th * 100)

        if scor >= th:
            couleur_delta = 'red'
        elif scor < th:
            couleur_delta = 'Orange'

        if scor >= th:
            valeur_delta = "red"
        elif scor < th:
            valeur_delta = "green"

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=scor,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Selected Customer Score", 'font': {'size': 25}},
            delta={'reference': int(th), 'increasing': {'color': valeur_delta}},
            gauge={
                'axis': {'range': [None, int(100)], 'tickwidth': 1.5, 'tickcolor': "black"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, int(th)], 'color': 'lightgreen'},
                    {'range': [int(th), int(scor)], 'color': couleur_delta}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 1,
                    'value': int(th)}}))

        fig.update_layout(paper_bgcolor="lavender", font={'color': "darkblue", 'family': "Arial"})
        return fig



    ##############################################################################
    #                         Customer's data checkbox
    ##############################################################################
    if st.sidebar.checkbox("Customer's data"):
        st.markdown('data of the selected customer :')
        data_selected_cust, y_cust = get_selected_cust_data(selected_id)
        # data_selected_cust.columns = data_selected_cust.columns.str.split('.').str[0]
        st.write(data_selected_cust)
    ##############################################################################
    #                         Model's decision checkbox
    ##############################################################################
    if st.sidebar.checkbox("Model's decision", key=38):
        # Get score & threshold model
        score, threshold_model = get_score_model(selected_id)
        # Display score (default probability)
        st.write('Default probability : {:.0f}%'.format(score * 100))
        # Display default threshold
        st.write('Default model threshold : {:.0f}%'.format(threshold_model * 100))  #
        # Compute decision according to the best threshold (False= loan accepted, True=loan refused)
        if score >= threshold_model:
            decision = "Loan rejected"
        else:
            decision = "Loan granted"
        st.write("Decision :", decision)
        ##########################################################################
        #              Display customer's gauge meter chart (checkbox)
        ##########################################################################
        figure = gauge_plot(score, threshold_model)
        st.write(figure)
        # Add markdown
        st.markdown('_Gauge meter plot for the applicant customer._')
        expander = st.expander("Concerning the classification model...")
        expander.write("The prediction was made using the Light Gradient Boosting classifier Model")
        expander.write("The default model is calculated to maximize air under ROC curve => maximize \
                                        True Positives rate (TP) detection and minimize False Negatives rate (FP)")
        ##########################################################################
        #                 Display local SHAP waterfall checkbox
        ##########################################################################
        if st.checkbox('Display waterfall local interpretation', key=25):
            with st.spinner('SHAP waterfall plots displaying in progress..... Please wait.......'):
                # Get Shap values for customer & expected values
                shap_vals, expected_vals = values_shap(selected_id)
                # index_cust = customer_ind(selected_id)
                # Get features names
                features = feat()
                # st.write(features)
                nb_features = st.slider("Number of features to display",
                                        min_value=2,
                                        max_value=10,
                                        value=10,
                                        step=None,
                                        format=None,
                                        key=14)
                # draw the waterfall graph (only for the customer with scaling
                waterfall_plot(nb_features, features, expected_vals[0][0], shap_vals.values)

                plt.gcf()
                st.pyplot(plt.gcf())
                # Add markdown
                st.markdown('_SHAP waterfall Plot for the applicant customer._')
                # Add details title
                expander = st.expander("Concerning the SHAP waterfall  plot...")
                # Add explanations
                expander.write("The above waterfall  plot displays \
                explanations for the individual prediction of the applicant customer.\
                The bottom of a waterfall plot starts as the expected value of the model output \
                (i.e. the value obtained if no information (features) were provided), and then \
                each row shows how the positive (red) or negative (blue) contribution of \
                each feature moves the value from the expected model output over the \
                background dataset to the model output for this prediction.")

        ##########################################################################
        #              Display feature's distribution (Boxplots)
        ##########################################################################
        if st.checkbox('show features distribution by class', key=20):
            st.header('Histplot of the main features')
            fig, ax = plt.subplots(figsize=(20, 10))
            with st.spinner('histplot creation in progress...please wait.....'):
                # Get Shap values for customer
                shap_vals, expected_vals = values_shap(selected_id)
                # Get features names
                features = feat()
                
	            #Age distribution plot
                data_all = get_all_cust_data()
                data_cust=get_selected_cust_data(selected_id)
                for f in features:
                	df_income = pd.DataFrame(data_all[f])
                	fig, ax = plt.subplots(figsize=(10, 5))
                	sns.histplot(df_income, edgecolor = 'k', color="goldenrod", bins=20)
                	ax.axvline(int(data_cust[f].values), color="green", linestyle='--')
                	ax.set(title='Distribution normal', xlabel=f, ylabel='')
                	st.pyplot(fig)

                st.markdown('_Dispersion of the main features for random sample,\
                20 nearest neighbors and applicant customer_')

                expander = st.expander("Concerning the dispersion graph...")
                expander.write("These boxplots show the dispersion of the preprocessed features values\
                used by the model to make a prediction. The green boxplot are for the customers that repaid \
                their loan, and red boxplots are for the customers that didn't repay it.Over the boxplots are\
                superimposed (markers) the values\
                of the features for the 20 nearest neighbors of the applicant customer in the training set. The \
                color of the markers indicate whether or not these neighbors repaid their loan. \
                Values for the applicant customer are superimposed in yellow.")


if __name__ == "__main__":
    main()
