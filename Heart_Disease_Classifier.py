import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from tkinter import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("heart.csv")
y = df['target'].values
X = df.drop(['target'], axis = 1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

lr_model = LogisticRegression(max_iter=100)
lr_model.fit(X_train, y_train)
lr_model.score(X_test, y_test)
Final_Model = lr_model

df = pd.read_csv("heart.csv")
df.head()
df.info()
df.describe()
df.target.value_counts()

sns.countplot(x="target", data=df, palette="bwr")
plt.show()

sns.countplot(x='sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()

pd.crosstab(df.sex, df.target).plot(kind="bar", figsize=(15, 6), color=['#1CA53B', '#AA1111'])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()

def check_inputs():
    if age.get() == "":
        print("Age Field is Empty!!")
        Label(win, text="Age Field is Empty!!", fg="blue", bg="yellow", font=("Calibri 10 bold")).place(x=12, y=580)
    elif rbp.get() == "":
        print("Resting Blood Pressure Field is Empty!!")
        Label(win, text="Resting Blood Pressure Field is Empty!!", fg="blue", bg="yellow", font=("Calibri 10 bold")).place(x=12, y=580)
    elif chol.get() == "":
        print("Cholestrol Field is Empty!!")
        Label(win, text="Cholestrol Field is Empty!!", fg="blue", bg="yellow", font=("Calibri 10 bold")).place(x=12, y=580)
    elif heart_rate.get() == "":
        print("Heart Rate Field is Empty!!")
        Label(win, text="Heart Rate Field is Empty!!", fg="blue", bg="yellow", font=("Calibri 10 bold")).place(x=12, y=580)
    elif peak.get() == "":
        print("Depression By Exercise Field is Empty!!")
        Label(win, text="Depression By Exercise Field is Empty!!", fg="blue", bg="yellow", font=("Calibri 10 bold")).place(x=12, y=580)
    else:
        predict()

def predict():
    gender_dict = {"Male": 1, "Female": 0}
    fbs_dict = {"True": 1, "False": 0}
    eia_dict = {"True": 1, "False": 0}
    cp_dict = {"1: typical angina": 0, "2: atypical angina": 1, "3: non-anginal pain": 2, "4: asymptomatic": 3}
    thal_dict = {"0: No Test": 0, "1: Fixed Defect": 1, "2: Normal Flow": 2, "3: Reversible Defect": 3}
    Pred_dict = {0: "Prediction: No Heart Disease Detected", 1: "Prediction: Heart Disease Detected\nYou should consult with your Doctor!"}
    
    data = [float(age.get()), gender_dict[str(radio.get())], cp_dict[str(variable.get())], float(rbp.get()),
            float(chol.get()), fbs_dict[str(radio_fbs.get())], int(str(variable_ecg.get())) - 1, float(heart_rate.get()),
            eia_dict[str(radio_eia.get())], float(peak.get()), int(str(variable_slope.get())) - 1, int(str(variable_n_vessels.get())) - 1,
            thal_dict[str(variable_thal.get())]]

    prediction = Final_Model.predict(np.array(data).reshape(1, 13))
    pred_label = Pred_dict[prediction.tolist()[0]]
    Label(win, text=pred_label, fg="blue", bg="yellow", font=("Calibri 10 bold")).place(x=12, y=580)

def reset():
    age.set("")
    rbp.set("")
    chol.set("")
    heart_rate.set("")
    peak.set("")

win = Tk()
win.geometry("450x600")
win.configure(background="#Eaedee")
win.title("Heart Disease Classifier")

title = Label(win, text="Heart Disease Classifier", bg="#2583be", width="300", height="2", fg="white", font=("Arial 20 italic")).pack()

Label(win, text="Age in Years", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=65)
Label(win, text="Resting Blood Pressure ", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=105)
Label(win, text="Cholestrol mg/dl ", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=145)
Label(win, text="Maximum Heart Rate ", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=185)
Label(win, text="Depression By Exercise ", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=225)
Label(win, text="Gender ", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=265)

radio = StringVar()
Radiobutton(win, text="Male", bg="#Eaedee", variable=radio, value="Male", font=("Verdana 12")).place(x=160, y=265)
Radiobutton(win, text="Female", bg="#Eaedee", variable=radio, value="Female", font=("Verdana 12")).place(x=260, y=265)

Label(win, text="Fasting Blood Pressure ", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=285)
radio_fbs = StringVar()
Radiobutton(win, text="True", bg="#Eaedee", variable=radio_fbs, value="True", font=("Verdana 12")).place(x=160, y=285)
Radiobutton(win, text="False", bg="#Eaedee", variable=radio_fbs, value="False", font=("Verdana 12")).place(x=260, y=285)

Label(win, text="Exercise Induced Angina", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=305)
radio_eia = StringVar()
Radiobutton(win, text="True", bg="#Eaedee", variable=radio_eia, value="True", font=("Verdana 12")).place(x=160, y=305)
Radiobutton(win, text="False", bg="#Eaedee", variable=radio_eia, value="False", font=("Verdana 12")).place(x=260, y=305)

Label(win, text="Chest Pain ", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=345)
variable = StringVar(win)
variable.set("CP")
OptionMenu(win, variable, "1: typical angina", "2: atypical angina", "3: non-anginal pain", "4: asymptomatic").place(x=140, y=345)

Label(win, text="Resting ECG ", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=385)
variable_ecg = StringVar(win)
variable_ecg.set("ECG")
OptionMenu(win, variable_ecg, "1", "2", "3").place(x=140, y=385)

Label(win, text="Slope of Exercise ", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=425)
variable_slope = StringVar(win)
variable_slope.set("Slope")
OptionMenu(win, variable_slope, "1", "2", "3").place(x=140, y=425)

Label(win, text="Thallium Stress ", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=465)
variable_thal = StringVar(win)
variable_thal.set("Thal")
OptionMenu(win, variable_thal, "0: No Test", "1: Fixed Defect", "2: Normal Flow", "3: Reversible Defect").place(x=140, y=465)

Label(win, text="Number Vessels ", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=505)
variable_n_vessels = StringVar(win)
variable_n_vessels.set("N_Vessels")
OptionMenu(win, variable_n_vessels, "1", "2", "3", "4").place(x=140, y=505)

age = StringVar()
rbp = StringVar()
chol = StringVar()
heart_rate = StringVar()
peak = StringVar()

Entry(win, textvariable=age, width=30).place(x=150, y=65)
Entry(win, textvariable=rbp, width=30).place(x=150, y=105)
Entry(win, textvariable=chol, width=30).place(x=150, y=145)
Entry(win, textvariable=heart_rate, width=30).place(x=150, y=185)
Entry(win, textvariable=peak, width=30).place(x=150, y=225)

Button(win, text="Reset", width="12", height="1", activebackground="red", command=reset, bg="Pink", font=("Calibri 12")).place(x=24, y=540)
Button(win, text="Classify", width="12", height="1", activebackground="violet", bg="Pink", command=check_inputs, font=("Calibri 12")).place(x=240, y=540)

win.mainloop()