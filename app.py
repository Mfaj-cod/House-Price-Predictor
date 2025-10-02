from flask import Flask, request, render_template
import pickle
import pandas as pd


with open('resources/model.pkl', 'rb') as m:
    model = pickle.load(m)

with open('resources/scaler.pkl', 'rb') as s:
    scaler = pickle.load(s)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        Square_Footage = request.form.get('Square_Footage')
        Num_Bedrooms = request.form.get('Num_Bedrooms')
        Year_Built = request.form.get('Year_Built')
        Lot_Size = request.form.get('Lot_Size')
        Garage_Size = request.form.get('Garage_Size')

        data = {
            'Square_Footage': Square_Footage,
            'Num_Bedrooms': Num_Bedrooms,
            'Year_Built': Year_Built,
            'Lot_Size': Lot_Size,
            'Garage_Size': Garage_Size
        }
        
        df = pd.DataFrame(data, index=[0], columns=['Square_Footage', 'Num_Bedrooms', 'Year_Built', 'Lot_Size', 'Garage_Size'])
        x_test = df.values
        x_test = scaler.transform(x_test)

        prediction = model.predict(x_test)

        return render_template('home.html', results=prediction[0].round(2))
    

if __name__=='__main__':
    app.run(debug=True)