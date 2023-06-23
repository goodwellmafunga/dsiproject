import streamlit as st
import plotly.graph_objects as go
import modelworker as mw 

# Define the currency options
currency_options = {
    'Euro': 'EUR',
    'Pound': 'GBP',
    'Canadian Dollar': 'CAD',
    'Swiss Franc': 'CHF',
    'Australian Dollar': 'AUD',
    'Yen': 'USDJPY=X'
}

# Define the value options
value_options = [ 3, 5, 7]

# Add a title to the app
st.title('Predictive Modelling of Currency Pairs')

# Create a sidebar for currency and value selection
with st.sidebar:
    currency_selected = st.selectbox('Select Currency:', list(currency_options.keys()))
    value_selected = st.selectbox('Number of Days:', value_options)
    
    data_dict = {'data': [], 'layout': {}}

    # Add a button to submit the form
    if st.button('Submit'):
        # Perform conversion based on selected options
        #converted_value = value_selected * conversion_rate(currency_options[currency_selected])
        #pass
        data_dict =mw.plotGraph(currency_selected,value_selected)
        # Display the result to the user
        #st.write(f'Converted value: {converted_value} {currency_selected}')


fig = go.Figure(data_dict['data'], data_dict['layout'])
st.plotly_chart(fig)